import math
import os
from recommenderSystem6.settings import MOVIELENS_ROOT

import pandas as pd


class NearestNeighbors:
    movieDF = pd.DataFrame()
    ratingsDF = pd.DataFrame()
    usersDF = pd.DataFrame()

    def __init__(self):
        try:
            moviesPath = os.path.join(MOVIELENS_ROOT, 'movies.dat')
            ratingsPath = os.path.join(MOVIELENS_ROOT, 'ratings.dat')
            usersPath = os.path.join(MOVIELENS_ROOT, 'users.dat')

            self.movieDF = pd.read_csv(moviesPath,
                                       sep="::",
                                       names=['MovieID', 'Title', 'Genres'],
                                       encoding='windows-1252',
                                       engine='python')

            self.ratingsDF = pd.read_csv(ratingsPath,
                                         sep="::",
                                         names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                                         encoding='windows-1252',
                                         engine='python').head(1799)

            self.usersDF = pd.read_csv(usersPath,
                                       sep="::",
                                       names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                                       encoding='windows-1252',
                                       engine='python').head(16)

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

    def getRatingForMovie(self, userId: int, movieId: int):
        ratingsDF = self.getMoviesTheUserRated(userId)
        movieDF = ratingsDF[self.ratingsDF['MovieID'] == movieId]
        if len(movieDF.index) > 0:
            return float(movieDF.iloc[0]['Rating'])
        else:
            return float('NaN')

    def predictRatingForMovieForUser(self, userId: int, movieId: int, n: int, simUsers: pd.DataFrame):
        # get average rating of current user
        avgA = self.getAverageRating(userId)
        # predict rating for movie
        sum1: float = 0
        sum2: float = 0
        for simU in simUsers:
            movieRating = self.getRatingForMovie(simU['id'], movieId)
            if not(math.isnan(simU['ratingAvg'])) and not(math.isnan(movieRating)):
                sum1 += simU['similarity'] * (movieRating - simU['ratingAvg'])
                sum2 += simU['similarity']
        if sum2 == 0:
            return 0
        pred: float = avgA + sum1 / sum2
        return pred

    def nearestNeighborRecommendation(self, userId: int, n: int = 10):
        # get similar users
        simUsers = self.computeSimilarity(userId, n)
        smallMovieDF = self.movieDF[['MovieID', 'Title', 'Genres']].copy()
        for index, row in smallMovieDF.iterrows():
            smallMovieDF.loc[smallMovieDF.index[index], 'PredictionScore'] = self.predictRatingForMovieForUser(userId, row['MovieID'], n, simUsers)
            # print(row)
        moviesSortedByPredictionScore = smallMovieDF[['Title', 'Genres', 'PredictionScore']].copy().sort_values(
            by=['PredictionScore'], ascending=[False])
        return moviesSortedByPredictionScore.head(20)

    def computeSimilarity(self, userID: int, neighborhoodSize: int):
        # Calculate the similarity of users with the pearson correlation
        # Returns - list of similarity values between the user and other users
        currentUserRated = self.getMoviesTheUserRated(userID)
        currentUserAverage = self.getAverageRating(userID)
        # List of all possible neighbors with their corresponding similarity value
        neighbors = []
        # Iterate through all users
        for index, otherUser in self.usersDF.iterrows():
            otherID = otherUser['UserID']
            # In order to skip the same user as similarity will always be high
            if userID == otherID:
                continue

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
                continue
            neighbors.append(dict(id=otherID, similarity=similarityValue, ratingAvg=otherAverageRating))

        nNearestNeighbors = sorted(neighbors, key=lambda item: item['similarity'], reverse=True)[:neighborhoodSize]
        return nNearestNeighbors
