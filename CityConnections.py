# Google is not able to plot real routes
# Install necessary libraries
# Use line below to install library, if it doesn't work
# !{sys.executable} -m pip install geopy
import sys
import webbrowser
import requests
import csv
import geopy.distance
import numpy
import googlemaps
import folium
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster

gmaps = googlemaps.Client(key='API')

# Creating dictionary of countires and their codes using our database
countries = {}
countries_keys = []
with open('CountryCodes.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        countries[row[1]] = row[0]
        countries_keys.append(row[1])

print("Please, enter the country name:")

# Checking the correctness of user's input
InputCountry = str(input())
i = 0
while (i != 1):
    if InputCountry in countries.keys():
        i = 1
        InputCountry = "&country=" + countries[InputCountry]
    elif (InputCountry == "c"):
        print(*countries_keys, sep=", ")
        print("\n")
        InputCountry = str(input())
    else:
        print("There is not such country, please, enter correct name.\
 If you would like to see the list of existing countries, enter 'c'.")
        InputCountry = str(input())

# Short names for parts of link
str1 = "http://api.geonames.org/searchJSON?&maxRows=1000"
str2 = "&featureClass=P&featureCodePPL&cities=cities15000&username=premium"


# First request
response = requests.get(str1 + InputCountry + str2)
city = response.json()
m = city['totalResultsCount']
if (m == 1):
    print("There is " + str(m) + " city.")
else:
    print("There are " + str(m) + " cities.")


k1 = []  # List of cities and their coordinates
if (m > 0):
    for p in city['geonames']:
        # print(p['toponymName'] + " " + p['lat'] + " " + p['lng'])
        k2 = []
        k2.extend([p['toponymName'], p['lat'], p['lng']])
        k1.append(k2)


# If there more than 1000 cities, send other requests
count = 1001
while (count < m) or (count < 5000):
    c = "&startRow=" + str(count)
    response = requests.get(str1 + c + InputCountry + str2)
    city = response.json()
    for p in city['geonames']:
        # print(p['toponymName'] + " " + p['lat'] + " " + p['lng'])
        k2 = []
        k2.extend([p['toponymName'], float(p['lat']), float(p['lng'])])
        k1.append(k2)
    count = count + 1000


# Create a CSV file with cities names and their coordinates
city_index = {}
with open("output.csv", 'a', encoding='utf-8') as outcsv:
    # configure writer to write standard CSV file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    index = 0
    for item in k1:
        # Write item to outcsv
        writer.writerow([item[0], item[1], item[2]])
        city_index[item[0]] = index
        index += 1
print("CSV file has been created.")


# Creating matrix of distances between cities and weighted graph
Graph = []
distances = numpy.zeros((len(k1), len(k1)))
for i in range(0, len(k1)):
    for j in range(i, len(k1)):
        coords_1 = (k1[i][1], k1[i][2])
        coords_2 = (k1[j][1], k1[j][2])
        total = geopy.distance.geodesic(coords_1, coords_2).km
        distances[i][j] = total
        distances[j][i] = total
        Graph.append([i, j, total])
print("Matrix of distances has been created.")


# Jarník's algorithm, Prim–Jarník algorithm, Prim–Dijkstra algorithm or the DJP algorithm
def PrimSpanningTree(V, G, DistanceMatrix):
    # Starting with zero vertex
    vertex = 0

    # Create empty arrays for algorithm
    MST = []
    edges = []
    visited = []
    minEdge = [None, None, float('inf')]

    # Repeating the algorithm until MST contains all vertices
    while len(MST) != V - 1:
        # mark this vertex as visited
        visited.append(vertex)

        # Edges that may be possible for connection
        for e in range(0, V):
            if DistanceMatrix[vertex][e] != 0:
                edges.append([vertex, e, DistanceMatrix[vertex][e]])

        # Find edge with the smallest weight for a vertex that is not visited
        for e in range(0, len(edges)):
            if edges[e][2] < minEdge[2] and edges[e][1] not in visited:
                minEdge = edges[e]

        edges.remove(minEdge)
        MST.append(minEdge)

        # start at new vertex and reset min edge
        vertex = minEdge[1]
        minEdge = [None, None, float('inf')]
    return MST


# Kruskal's algorithm, V is number of vertices
def KruskalSpanningTree(V, Graph):
    # Sort edges in graph by their weigth
    Graph.sort(key=lambda x: x[2])
    result = []
    empty = set()

    # Create set for each vertice
    vertices = {}
    for i in range(0, V):
        vertices[i] = set([i])

    for edge in range(0, len(Graph)):
        begin = Graph[edge][0]
        end = Graph[edge][1]
        if (vertices[begin].intersection(vertices[end]) == empty):
            result.append([begin, end])
            temporary = vertices[begin].union(vertices[end])
            vertices[begin] = temporary
            vertices[end] = temporary
            for vertice in vertices[end]:
                vertices[vertice] = temporary
    return result


# Boruvka's way of solving problems
def Boruvka(distances):
    setMatrix = []
    allEdges = []
    for i in range(0, len(distances)):
        setMatrix.append([i])

    def combine(e):
        e0 = -1
        e1 = -1
        for i in range(0, len(setMatrix)):
            if e[0] in setMatrix[i]:
                e0 = i
            if e[1] in setMatrix[i]:
                e1 = i
        setMatrix[e0] += setMatrix[e1]
        del setMatrix[e1]

    while (len(setMatrix) > 1):
        edges = []
        for component in setMatrix:
            m = [9999999, [0, 0]]
            for vertex in component:
                for i in range(0, len(distances[0])):
                    if i not in component and distances[vertex][i] != 0:
                        if (m[0] > distances[vertex][i]):
                            m[0] = distances[vertex][i]
                            m[1] = [vertex, i]
            if (m[1][0] > m[1][1]):
                m[1][0], m[1][1] = m[1][1], m[1][0]
            if (m[1] not in edges):
                edges.append(m[1])
        for e in edges:
            combine(e)
            allEdges.append(e)
    return allEdges


# Create map
the_map = folium.Map(location=[k1[0][1], k1[0][2]], zoom_start=5)

# Create Cluster
marker_cluster = MarkerCluster().add_to(the_map)
for i in range(len(k1)):
    folium.Marker(location=[k1[i][1], k1[i][2]], popup=k1[i][0], icon=folium.Icon(color='gray')).add_to(marker_cluster)

# Connect all cities using minimum spanning tree
# Chose one of your preference
print("Choose algorithm for connecting cities:")
print("1 - Jarník's algorithm, Prim–Jarník algorithm, Prim–Dijkstra algorithm.")
print("2 - Kruskal's algorithm")
print("3 - Boruvka's algorithm ")
print("Enter number of algorhitm. If input is incorrect, program will choose the first one.")
AlgorithmChoice = input()
if (AlgorithmChoice == "2"):
    roads = KruskalSpanningTree(len(k1), Graph)
elif (AlgorithmChoice == "3"):
    roads = Boruvka(distances)
else:
    roads = PrimSpanningTree(len(k1), Graph, distances)

# Sets for charts
n_groups = len(roads)
im_dist = []
r_dist = []
im_time = []
r_time = []
names = []
for el in roads:
    names.append(str(k1[el[0]][0] + '\n' + k1[el[1]][0]))

# City connection
for i in range(0, len(roads)):
    coords_1 = (float(k1[roads[i][0]][1]), float(k1[roads[i][0]][2]))
    coords_2 = (float(k1[roads[i][1]][1]), float(k1[roads[i][1]][2]))
    line = [coords_1, coords_2]
    length = distances[roads[i][0]][roads[i][1]]
    im_dist.append(length)
    if length > 99:
        length = int(length)
    else:
        length = "%.2f" % distances[roads[i][0]][roads[i][1]]
    dist = str(length) + " km"
    hours = int(distances[roads[i][0]][roads[i][1]]//70)
    minutes = int((distances[roads[i][0]][roads[i][1]] % 70 * 0.6))
    time = ""
    im_time.append(60*hours + minutes)
    if hours > 0:
        time += str(hours) + " hours "
    if minutes > 0:
        time += str(minutes) + " minutes"
    folium.PolyLine(locations=line, weight=5, color='green', tooltip="distance: " + dist + ";  time: " + time).add_to(the_map)

    # Inconvenient google maps
    directions = gmaps.directions(coords_1, coords_2)
    if(directions):
        start_lat = directions[0]['legs'][0]['steps'][0]['start_location']['lat']
        start_lng = directions[0]['legs'][0]['steps'][0]['start_location']['lng']
        start_point = (float(start_lat), float(start_lng))
        actual_time = directions[0]['legs'][0]['duration']['text']
        comlicated_time = 0
        if actual_time.split()[1] == 'hours':
            comlicated_time = int(actual_time.split()[0]) * 60 + int(actual_time.split()[2])
        else:
            comlicated_time = int(actual_time.split()[0])
        r_time.append(comlicated_time)
        actual_distance = directions[0]['legs'][0]['distance']['text']
        r_dist.append(float(actual_distance.split()[0]))

        longline = []
        for p in directions[0]['legs'][0]['steps']:
            end_lat = p['end_location']['lat']
            end_lng = p['end_location']['lng']
            end_point = (float(end_lat), float(end_lng))
            longline.append([start_point, end_point])
            start_point = end_point

        points_on_roads = []

        for el in longline:
            # print(str(el) + "orig")
            res = gmaps.snap_to_roads(el, True)
            for k in res:
                point_lat = k['location']['latitude']
                point_lng = k['location']['longitude']
                points_on_roads.append([point_lat, point_lng])
                # print([point_lat, point_lng])
            # print()

        start = points_on_roads[0]
        for el in points_on_roads:
            info = "actual distance: " + actual_distance + ";    time: " + actual_time
            folium.PolyLine(locations=[start, el], weight=3, color='dodgerblue',
                            tooltip=info).add_to(the_map)
            start = el

# Add layer control
folium.TileLayer('openstreetmap').add_to(the_map)
folium.TileLayer('stamenterrain').add_to(the_map)
folium.TileLayer('CartoDB dark_matter').add_to(the_map)
folium.LayerControl().add_to(the_map)

# create plot for dist
plt.figure(1)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, r_dist, bar_width,
alpha=opacity,
color='b',
label='Road distances')

rects2 = plt.bar(index + bar_width, im_dist, bar_width,
alpha=opacity,
color='g',
label='Geometric distances')

plt.xlabel('Roads')
plt.ylabel('Km')
plt.title('Distances compared')
plt.xticks(index + bar_width, names)
plt.legend()

plt.tight_layout()

# create plot for time
plt.figure(2)
fig2, ax2 = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, r_time, bar_width,
alpha=opacity,
color='b',
label='Actual time')

rects2 = plt.bar(index + bar_width, im_time, bar_width,
alpha=opacity,
color='g',
label='Approximated time')

plt.xlabel('Roads')
plt.ylabel('Minutes')
plt.title('Times compared')
plt.xticks(index + bar_width, names)
plt.legend()

plt.tight_layout()
plt.show()

# Save map
the_map.save("map.html")
print("Map has been saved.")
webbrowser.open('map.html')

