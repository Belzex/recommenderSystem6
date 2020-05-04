from django.shortcuts import render

# Create your views here.
from testWebsite.NearestNeighbors import NearestNeighbors

nn = NearestNeighbors()

def home(request):
    if request.method == 'POST':
        data = request.POST.copy()
        id = data.get('idUserField')
        id = int(id)
        list = nn.nearestNeighborRecommendation(id)
        meta_dict = nn.getMetaData(list)
        print(meta_dict)
        dict = list.to_dict('records')
        print(dict)
        i = 0
        for d in dict:
            a = meta_dict[i]
            if len(a) > 0:
                d.update(a[0])
            i += 1
        return render(request, "recommendationPage.html", {'id': id, 'list': dict})
    return render(request, "mainPage.html", {})


def recommendation(request):
    return render(request, "recommendationPage.html", {})
