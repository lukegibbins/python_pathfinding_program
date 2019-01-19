
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from random import *


# 2D 12x12 grid
grid = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])


# start point and end point (tuples)
start = (0, 1)
end = (8, 6)


# heuristic for path scoring. Tries to find the smallest cost between nodes
# accepts 2 nodes (tupleA, tupleB) where 'tupleA' is the current tuple and 'tupleB' is the intended tuple
# to calculate the cost. This can be done for movement cost g(n) or cost from current to end h(n).
# the method is used twice for this operations.
def heuristic(tupleA, tupleB):
    return np.sqrt((tupleB[0] - tupleA[0]) ** 2 + (tupleB[1] - tupleA[1]) ** 2)


# astar algorithm
def astar(grid, start, end):

    # list of possible tuples the current position can view and compare
    # possible directions: up, up/right, right, down/right, down, down/left, left, up/left
    neighbouring_tuples = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # list used to hold visited positions or tuples. After each traversal, the tuple or nodes visited will be
    # appended to this list
    closed_list = set()

    # contains  all of the routes/paths taken in each iteration during traversal
    # when the end point is found, the shortest path is taken from this object
    came_from = {}

    # used as a variable to sore the G score in the astar algorithm: f(n) = g(n) + h(n)
    # has to be initialised to 0.
    gscore = {start: 0}

    # used as a variable to store the F score
    # straight away we know the F score using the start and end coordinates defined at run time
    # to calculate the F score
    fscore = {start: heuristic(start, end)}

    # list of positions that are being considered for the shortest path. If the end goal is found on
    # the first path, the open_list will be empty.
    open_list = []

    # pushes fscore (being the cost from current start to current end) and the starting position
    # into the open_list as a value. E.g -->  '0.0 (1,1)'
    # the open list will always prioritised the lowest value to the top
    heapq.heappush(open_list, (fscore[start], start))

    # while open_list has a value
    # check for all neighbouring_tuples a node can visit until there are no options left
    while open_list:

        # extract out the current tuple from the list. e.g (1,1)
        # the current_tuple will be evaluated after each while loop and will
        # change its value when a new short path has been found
        current_tuple = heapq.heappop(open_list)[1]

        # if the current tuple is equal to the end tuples coordinates,
        # extract and return the shortest path to the filtered_path_data
        if current_tuple == end:
            # defines empty array to hold the lowest cost nodes after being evaluated
            filtered_path_data = []
            # while the current_tuple is in came_from
            while current_tuple in came_from:
                # append the tuples to the list
                filtered_path_data.append(current_tuple)
                current_tuple = came_from[current_tuple]
            return filtered_path_data

        # if the current node has not reached the end, add the visited current node to the closed_list.
        # We do not need to re-evaluate the current node anymore.
        closed_list.add(current_tuple)

        # for each neighbouring tuple element(0) and tuple element(1) in the neighbouring_tuples list
        # this will loop through each neighbouring node while still in the while loop for the current
        # item in the open_list as evaluate each neighbouring tuples cost using tentative_g_score
        for tupleElement0, tupleElement1 in neighbouring_tuples:

            # get the current node and add the values from the current node to the first neighbouring node
            # to produce a current_neighbour_tuple
            current_neighbour_tuple = current_tuple[0] + tupleElement0, current_tuple[1] + tupleElement1

            # calculate the tentative_g_score as a cost to each neighbouring tuple
            # it is called 'tentative' as the gscore will only be settled when all costs have been
            # calculated from neighbouring nodes and the lowest cost has been found.
            # the lowest tentative gscore will be come the gscore.
            tentative_g_score = gscore[current_tuple] + heuristic(current_tuple, current_neighbour_tuple)

            # if the neighbour is outside the grid or not reachable, ignore the current neighbour
            # and continue to with the loop
            if 0 <= current_neighbour_tuple[0] < grid.shape[0]:
                if 0 <= current_neighbour_tuple[1] < grid.shape[1]:
                    # if there is a obstacle
                    if grid[current_neighbour_tuple[0]][current_neighbour_tuple[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            # if the neighbour is in the closed set or has a greater tentative_g_score has a greater score
            # than its current position, simply ignore and continue
            if current_neighbour_tuple in closed_list and tentative_g_score >= gscore.get(current_neighbour_tuple, 0):
                continue

            # if the tentative__g_score is less than the current gscore (has a lower score)
            # or the neighbour being evaluated is not in the open list, then update and add it to the open list.
            # and update each value
            if tentative_g_score < gscore.get(current_neighbour_tuple, 0) or current_neighbour_tuple not in [i[1] for i in open_list]:

                # adds the current_tuple which has been evaluated to the came_from list
                # current_tuple is now updated with the position of the lowest score
                # and repeats the process to evaluate surrounding neighbours with the new position as the current tuple
                came_from[current_neighbour_tuple] = current_tuple

                # assign a new movement cost for gscore
                gscore[current_neighbour_tuple] = tentative_g_score

                # hscore used to calculate distance from current to end
                hscore = heuristic(current_neighbour_tuple, end)

                # fscore = tentative_g_score + hscore (we now have an established fscore)
                fscore[current_neighbour_tuple] = tentative_g_score + hscore

                # adds the current_neighbour_tuple to the open list with its fscore
                # on each iteration, nodes are added to the open_list
                heapq.heappush(open_list, (fscore[current_neighbour_tuple], current_neighbour_tuple))

    return False


# display the static path to the user that is defined before te program is executed
def _static_path(event):
    plt.close()
    create_graph_and_path(grid, start, end)


# display a random path to the user
def _rand_path(event):
    # while the path is random
    valid_random = False

    while valid_random is False:
        # keep finding a random value
        start_random_y = randint(0, 11)
        start_random_x = randint(0, 11)

        end_random_y = randint(0, 11)
        end_random_x = randint(0, 11)

        new_start = (start_random_y, start_random_x)
        new_end = (end_random_y, end_random_x)

        valid_path = (grid[new_start[0]][new_start[1]], grid[new_end[0]][new_end[1]])

        # if the path is not equal to a 1 or an obstacle is present
        # refresh graph with the random route
        if valid_path[0] != 1 and valid_path[1] != 1:
            valid_random = True
            plt.close()
            create_graph_and_path(grid, new_start, new_end)


# method to create the path and graph
def create_graph_and_path(grid, start, end):

    # path contains a list of tuples with the path from the end point to the start point,
    # and then add the start position to the list
    path = astar(grid, start, end)
    path = path + [start]

    # returns the list in reverse from end position to display to the user. Now start to end
    route = path[::-1]
    # prints out route
    print("\nPath from start to end node.")
    print(route)
    # extract x[0] and y[1] coordinates from route list
    x_coords = []
    y_coords = []

    # until the end of the list
    for i in (range(0, len(route))):
        # store column values in x and row values in y
        x = route[i][0]
        y = route[i][1]

        # add each separate tuple [0] & [1] to the list of coordinates to plot on the graph
        x_coords.append(x)
        y_coords.append(y)

    # plot map and path
    # define figure size
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=plt.cm.bone)

    # set start and end positions
    ax.scatter(start[1], start[0], marker='o', color="red", s=200)
    ax.scatter(end[1], end[0], marker="*", color="gold", s=250)

    # plot the path that has been found using x_coords and y_coords and display the figure
    ax.plot(y_coords, x_coords, color="yellow")
    plt.ylabel("Array No.")
    plt.xlabel("Element No.")
    plt.title("Pizza Delivery Route")

    # defines buttons on graph for static route which is set manually
    axcut = plt.axes([0.89, 0.01, 0.1, 0.045])
    bcut = Button(axcut, 'Static')
    bcut.on_clicked(_static_path)

    # defines buttons on graph for random path that is randomized
    axcut = plt.axes([0.74, 0.01, 0.123, 0.045])
    ccut = Button(axcut, 'Random')
    ccut.on_clicked(_rand_path)
    plt.show()


# calls create_Graph_And_Path() method and creates graph
create_graph_and_path(grid, start, end)










