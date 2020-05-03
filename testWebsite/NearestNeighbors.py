from recommenderSystem6.settings import MOVIELENS_ROOT
import math
import pandas as pd
import concurrent.futures
import csv
import ast
import os

MOVIE_DF_COLUMNS: list = ['MovieID', 'Title', 'Genres']
RATINGS_DF_COLUMNS: list = ['UserID', 'MovieID', 'Rating', 'Timestamp']
USER_DF_COLUMNS: list = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']


class NearestNeighbors:
    movieDF = pd.DataFrame(columns=MOVIE_DF_COLUMNS)
    ratingsDF = pd.DataFrame(columns=RATINGS_DF_COLUMNS)
    usersDF = pd.DataFrame(columns=USER_DF_COLUMNS)
    simUsersDict = dict()

    def __init__(self, userDF: pd.DataFrame = None, ratingsDF: pd.DataFrame = None, movieDF: pd.DataFrame = None):
        moviesPath = os.path.join(MOVIELENS_ROOT, 'movies.dat')
        ratingsPath = os.path.join(MOVIELENS_ROOT, 'ratings.dat')
        usersPath = os.path.join(MOVIELENS_ROOT, 'users.dat')
        self.simUsersDict = self.read()
        try:
            if userDF is None:
                self.usersDF = pd.read_csv(usersPath,
                                           sep="::",
                                           names=USER_DF_COLUMNS,
                                           encoding='windows-1252',
                                           engine='python').sample(frac=1)
            else:
                self.usersDF = userDF
            if ratingsDF is None:
                self.ratingsDF = pd.read_csv(ratingsPath,
                                             sep="::",
                                             names=RATINGS_DF_COLUMNS,
                                             encoding='windows-1252',
                                             engine='python').head(1799)
            else:
                self.ratingsDF = ratingsDF
            if movieDF is None:
                self.movieDF = pd.read_csv(moviesPath,
                                           sep="::",
                                           names=MOVIE_DF_COLUMNS,
                                           encoding='windows-1252',
                                           engine='python').head(16)
            else:
                self.movieDF = movieDF

        except FileNotFoundError:
            print("Could not find file")

        except IOError:
            print("Could not open file")

    def getMoviesTheUserRated(self, userId: int):
        # select the user
        currentUserDF = self.usersDF[self.usersDF['UserID'] == userId]
        # merge tables to contain user information and review to every movie rated by the user
        mergedUsersRatings = self.ratingsDF.merge(currentUserDF, left_on='UserID', right_on='UserID')
        mergedDF = self.movieDF.merge(mergedUsersRatings, left_on='MovieID', right_on='MovieID')
        return mergedDF

    def printMoviesTheUserRated(self, userId: int):
        mergedDF = self.getMoviesTheUserRated(userId)
        # shrink to Title and Genres
        smallMovieDF = mergedDF[['Title', 'Genres']].copy()
        # print the first 15 (or fewer) rated movies
        print("Movies rated by user with ID:", userId)
        print(self.getAverageRating(userId))
        print(smallMovieDF.head(15))

    def getAverageRating(self, userId: int):
        ratingsDF = self.getMoviesTheUserRated(userId)
        avg = ratingsDF['Rating'].mean()
        return avg

    def getRatingForMovie(self, userId: int, movieId: int) -> float:
        ratingsDF = self.getMoviesTheUserRated(userId)
        ratingsDF.set_index('MovieID')
        movieDF = ratingsDF.query('MovieID' + ' == ' + str(movieId))
        if len(movieDF.index) > 0:
            return float(movieDF.iloc[0]['Rating'])
        else:
            return float('NaN')

    def predictRatingForMovieForUser(self, userId: int, movieId: int, simUsers: pd.DataFrame) -> float:
        # get average rating of current user
        avgA = self.getAverageRating(userId)
        # predict rating for movie
        sum1: float = 0
        sum2: float = 0
        for simU in simUsers:
            movieRating = self.getRatingForMovie(simU['id'], movieId)
            if not (math.isnan(simU['ratingAvg'])) and not (math.isnan(movieRating)):
                sum1 += simU['similarity'] * (movieRating - simU['ratingAvg'])
                sum2 += simU['similarity']
        if sum2 == 0:
            return 0.0
        pred: float = avgA + sum1 / sum2
        return pred

    def nearestNeighborRecommendation(self, userId: int, n: int = 10):
        # get similar users
        simUsers = self.computeSimilarity(userId, n)
        smallMovieDF = self.movieDF[['MovieID', 'Title', 'Genres']].copy()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_predict_rating_for_movie_for_user = {
                executor.submit(self.predictRatingForMovieForUser, userId, row['MovieID'], simUsers): index for
                index, row in smallMovieDF.iterrows()}
            for future in concurrent.futures.as_completed(future_predict_rating_for_movie_for_user):
                ind = future_predict_rating_for_movie_for_user[future]
                try:
                    data = future.result()
                    smallMovieDF.loc[ind, 'PredictionScore'] = data
                except Exception as exc:
                    print('exception: %s' % (exc))
        moviesSortedByPredictionScore = smallMovieDF[['Title', 'Genres', 'PredictionScore']].copy().sort_values(
            by=['PredictionScore'], ascending=[False])
        return moviesSortedByPredictionScore.head(20)

    def computeSimilarity(self, userID: int, neighborhoodSize: int):
        if userID in self.simUsersDict:
            return self.simUsersDict[userID][:neighborhoodSize]
        print('fail')
        # Calculate the similarity of users with the pearson correlation
        # Returns - list of similarity values between the user and other users
        currentUserRated = self.getMoviesTheUserRated(userID)
        currentUserAverage = self.getAverageRating(userID)
        # List of all possible neighbors with their corresponding similarity value
        neighbors = []

        # Iterate through all users
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_m = {executor.submit(self.m, otherUser, userID, currentUserRated, currentUserAverage): index for
                        index, otherUser in self.usersDF.iterrows()}
            for future in concurrent.futures.as_completed(future_m):
                index = future_m[future]
                try:
                    data = future.result()
                    if data is not None:
                        neighbors.append(data)
                except Exception as exc:
                    print('exception: %s' % (exc))
        nNearestNeighbors = sorted(neighbors, key=lambda item: item['similarity'], reverse=True)
        self.simUsersDict[userID] = nNearestNeighbors
        self.store()
        nNearestNeighbors = nNearestNeighbors[:neighborhoodSize]
        return nNearestNeighbors

    def m(self, otherUser, userID, currentUserRated, currentUserAverage):
        otherID = otherUser['UserID']
        # In order to skip the same user as similarity will always be high
        if userID == otherID:
            return

        otherAverageRating = self.getAverageRating(otherID)
        otherRating = self.getMoviesTheUserRated(otherID)
        numerator: float = 0
        denominator1: float = 0
        denominator2: float = 0

        # Find the movies both users rated
        for currentMovieID in currentUserRated['MovieID']:
            for otherMovieID in otherRating['MovieID']:
                if currentMovieID == otherMovieID:
                    # Check for number in ratings
                    tempUserRating = self.getRatingForMovie(userID, currentMovieID)
                    tempOtherRating = self.getRatingForMovie(otherID, currentMovieID)
                    if not (math.isnan(tempUserRating) or math.isnan(tempOtherRating)):
                        # Calculate the necessary values
                        currentUserValue = tempUserRating - currentUserAverage
                        otherValue = tempOtherRating - otherAverageRating
                        numerator += currentUserValue * otherValue
                        denominator1 += math.pow(currentUserValue, 2)
                        denominator2 += math.pow(otherValue, 2)
        if not (denominator1 == 0 or denominator2 == 0):
            # Calculate the final similarity value for said user
            similarityValue: float = numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            # Calculation not possible - Skip value
            return
        return dict(id=otherID, similarity=similarityValue, ratingAvg=otherAverageRating)

    def store(self):
        w = csv.writer(open(os.path.join(MOVIELENS_ROOT, 'simUsers.csv'), "w"))
        for key, val in self.simUsersDict.items():
            w.writerow([key, val])

    def read(self):
        d = {}
        with open(os.path.join(MOVIELENS_ROOT, 'simUsers.csv')) as f:
            for line in f:
                key , val = line.split(",", 1)
                val=val.strip('"')
                val=ast.literal_eval(val)
                d[int(key)] = val
        return d

