from django.shortcuts import render

# Create your views here.
from testWebsite.NearestNeighbors import NearestNeighbors


def home(request):
    if request.method == 'POST':
        data = request.POST.copy()
        id = data.get('idUserField')
        id=int(id)
        nn = NearestNeighbors()
        list = nn.nearestNeighborRecommendation(id)
        dict = list.to_dict('records')
        print(dict)
        return render(request, "recommendationPage.html", {'id': id, 'list': dict})
    return render(request, "mainPage.html", {})


def recommendation(request):
    return render(request, "recommendationPage.html", {})
