"""
                                                    Marco Mendez
                                                    COMPSCI 216, Spring 2020
                                                    June 12, 2020

                        COMPSCI 216 Project

File Name:      maze_solver.py
Description:    This program solves a given maze provided by the user. The program
                reviews the image file specified by the user. From there, the image
                is transformed to a binar image consisting of only black and white
                pixels. Once transformed, it prompts the user to dictate the start
                point and end point of the maze. In addition, the user can draw
                additional borders. Finally, the user selects which method to use
                in order to solve the maze. From there, the program visually displays
                how the solution is being determined and tells the user if a path from
                the start point to the end point can be found.
"""


from collections import deque
from math import hypot
from queue import PriorityQueue
import colorsys, cv2, threading, sys


#Checks if user has provided sufficient arguments to run program
if len(sys.argv) < 2:
    print("Incorrect arguments provided")
    print("Please ensure that only one other argument is given")
    print("The second argument should be the image file of your maze you would like to solve")
    exit()


"""
Class:          Point
Description:    An abstraction used to represent pixels in an image
Data Fields:    x       -  The x-coordinate value of the point in the image
                y       -  The y-coordinate value of the point in the image
                visited -  A flag indicating if the point has been visited in a search
                value   -  A value used in tracking the points visited within the image
                parent  -  Indicates the parent of the point
                           Used when determining the path taken and the points used
                f       -  The cost of the node in determining the path solution
                h       -  Heuristic to determine distance from current point to end point
                name    -  A naming convention given to a created cell
Functions:      init    -  Defines the abstraction created
                add     -  Determines how two abstractions are added together
                eq      -  Determines how two abostractions are compared with one another
"""
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.value = 0
        self.parent = None
        self.f = 0
        self.h = 0
        self.name = y * width + x

    def __add__(self, other):
        sumX = self.x + other.x
        sumY = self.y + other.y
        return Point(sumX, sumY)

    def __eq__(self, other):
        compareX = self.x == other.x
        compareY = self.y == other.y
        return compareX and compareY


#Global variables used throughout maze solver program
#Tracks click actions and draw actions performed by user
numClicks = 0
startPoint = None
endPoint = None
windowOpen = True
drawMode = False
solutionFound = False
drawX, drawY = -1, -1


"""
Function:       clickEvent
Description:    This function is used to record the event of the user clicking the
                image being displayed
Result:         Plots a start point and end point onto the maze being displayed to
                the user
"""
def clickEvent(event, x, y, flags, param):

    global startPoint, endPoint, numClicks, drawMode, drawX, drawY

    #Checks if user clicks down on left mouse button
    if event == cv2.EVENT_LBUTTONDOWN:

        #Checks if user is performing first mouse click
        if numClicks == 0:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            print("Start point selected")
            startPoint = Point(x, y)
            numClicks += 1

        #Checks if user is performing second mouse click
        elif numClicks == 1:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            print("End point selected")
            endPoint = Point(x, y)
            numClicks += 1
            drawMode = True

        #Checks if the user is able to draw additional borders
        elif drawMode == True:
                drawX, drawY = x, y
                numClicks += 1

    #Checks if user releases left mouse button
    elif event == cv2.EVENT_LBUTTONUP:

        #Checks if user is able to draw additional borders
        if drawMode == True and numClicks > 2:
            cv2.line(img, (drawX, drawY), (x, y), (0, 0, 0), 2)


"""
Function:       disp
Description:    This function is used to display the maze provided by the user.
                This function is run through a thread in order to render the
                image as the solution is being determined for the maze.
Input:          title   -  The title given to the window being rendered
                maze    -  The maze image being rendered
Result:         Displays image to user in order for them to select start and end
                points on image. In addition, image is updated to reflect how the
                solution is being determined (based on the method selected by user).
"""
def disp(title, maze):

    global windowOpen, drawMode

    #Displays maze image and sets handler for mouse events
    cv2.imshow(title, maze)
    cv2.setMouseCallback(title, clickEvent)

    while windowOpen == True:
        cv2.imshow(title, maze)

        #Stops loop when user hits 'ESC' key
        if cv2.waitKey(1) == 27:
            windowOpen = False
            exit()

        #Updates flag when users hits 'SPACE' key
        if cv2.waitKey(1) == 32 and drawMode == True:
            drawMode = False


"""
Function:       getNeighbors
Description:    This function grabs the four surrounding neighbors of a cell (point)
                within an image. These neighbors are to the North, South, East, and
                West directions of the given point.
Input:          point  -  A point on the maze image being rendered
Result:         Returns a deque of the eight surrounding neighbors of the point
"""
def getNeighbors(point):

    pointDirections = [Point(-1, 0), Point(0, -1), Point(0, 1), Point(1, 0)]
    neighbors = deque()

    for dir in pointDirections:
        cell = point + dir
        neighbors.append(cell)

    return neighbors


"""
Function:       checkValidCell
Description:    This function checks if a particular cell (point) can be considered
                for use when determining the solution path. This validity is determined
                based on the location of the cell and the BGR values of the cell.
Input:          point   -   A point on the maze image being rendered
Result:         Returns True/False based on whether or not the point can be used in
                determining solution path of the maze
"""
def checkValidCell(point):

    #Checks if point location is within maze image
    if(point.x >= 0 and point.x < width and point.y >= 0 and point.y < height):

        #Checks if BGR values of point are not values normally associated to a border of the maze image
        if(img[point.y][point.x][0] != 0 or img[point.y][point.x][1] != 0 or img[point.y][point.x][2] != 0):
            return True
    
    return False


"""
Function:
Description:    This function returns the BGR value that a cell should be colored
                due to the point being traversed by the solver
Input:          value   -  THe value used to determine BGR color calculation
                factor  -  An underlying value used to evenly distribute color traversal
Result:         Returns the BGR value the cell is to be colored
"""
def colorTraversal(value, factor):

    r, g, b = colorsys.hsv_to_rgb((value / factor), 1, 1)

    r = r * 255
    g = g * 255
    b = b * 255

    return [b, g, r]


"""
Function:       BFS
Description:    This function performs a Breadth First Search (BFS) of the maze image
                provided by the user. Based on the search done, it will determine
                whether a solution of the maze can be found based on the start and
                end points indicated by the user.
Input:          start   -   The start point indicated by the user within the maze
                end     -   The end point indicated by the user within the maze
                maze    -   The image containing the maze provided by the user
                points  -   A grid of points representing every cell within the image
Result:         If a solution is found for the maze, calls a function to display the
                solution found. Otherwise, indicates to the user that no solution could
                be found for points indicated.
"""
def BFS(start, end, maze, points):

    queue = deque()

    #Updates values of start point in relation to grid of points
    #Adds start point to queue
    points[start.y][start.x].visited = True
    queue.append(points[start.y][start.x])

    while len(queue) > 0:

        #Pops the first item within the queue and grabs neighbors of point
        currPoint = queue.popleft()
        neighbors = getNeighbors(currPoint)

        for cell in neighbors:

            #Checks if neighboring cell can be used in path solution calculation
            if (checkValidCell(cell) == True and points[cell.y][cell.x].visited == False):

                #Updates value of neighboring cell in relation to grid of points
                #Adds neighboring cell to queue
                points[cell.y][cell.x].visited = True
                points[cell.y][cell.x].value = currPoint.value + 1
                queue.append(points[cell.y][cell.x])

                #Update visited cells in image to visualize BFS algorithm
                maze[cell.y][cell.x] = colorTraversal(points[cell.y][cell.x].value, 5000)

                #Updates parent value of neighboring cell
                points[cell.y][cell.x].parent = currPoint

                #Checks if neighboring cell is the end point specified by user
                if cell == end:
                    queue.clear()
                    break

    #Checks if end point specified by user was visited during search
    if points[end.y][end.x].visited == True:
        displaySolution(points[start.y][start.x], points[end.y][end.x], imgcopy, points)
    else:
        print("Path Not Found")


"""
Function:       DFS
Description:    This function performs a Depth First Search (DFS) of the maze image
                provided by the user. Based on the search done, it will determine
                whether a solution of the maze can be found based on the start and
                end points indicated by the user.
Input:          start   -   The start point indicated by the user within the maze
                end     -   The end point indicated by the user within the maze
                maze    -   The image containing the maze provided by the user
                points  -   A grid of points representing every cell within the image
Result:         If a solution is found for the maze, calls a function to display the
                solution found. Otherwise, indicates to user that no solution could
                be found for points indicated.
"""
def DFS(start, end, maze, points):

    stack = deque()

    #Updates values of start point in relation to grid of points
    #Adds start point to stack
    points[start.y][start.x].visited = True
    stack.append(points[start.y][start.x])

    while len(stack) > 0:

        #Pops the first item within the stack and grabs neighbors of point
        currPoint = stack.pop()
        neighbors = getNeighbors(currPoint)

        for cell in neighbors:

            #Checks if neighboring cell can be used in path solution calculation
            if (checkValidCell(cell) == True and points[cell.y][cell.x].visited == False):

                #Updates value of neighboring cell in relation to grid of points
                #Adds neighboring cell to stack
                points[cell.y][cell.x].visited = True
                points[cell.y][cell.x].value = currPoint.value + 1
                stack.append(points[cell.y][cell.x])

                #Update visited cells in image to visualize DFS algorithm
                maze[cell.y][cell.x] = colorTraversal(points[cell.y][cell.x].value, 35000)

                #Updates parent value of neighboring cell
                points[cell.y][cell.x].parent = currPoint

                #Checks if neighboring cell is the end point specified by user
                if cell == end:
                    stack.clear()
                    break

    #Checks if end point specified by user was visited during search
    if points[end.y][end.x].visited == True:
        displaySolution(points[start.y][start.x], points[end.y][end.x], imgcopy, points)
    else:
        print("Path Not Found")


"""
Function:       AStar
Description:    This function performs a A Star (A*) algorithm search of the maze image
                provided by the user. Based on the search done, it will determine
                whether a solution of the maze can be found based on the start and
                end points indicated by the user.
Input:          start   -   The start point indicated by the user within the maze
                end     -   The end point indicated by the user within the maze
                maze    -   The image containing the maze provided by the user
                points  -   A grid of points representing every cell within the image
Result:         If a solution is found for the maze, calls a function to display the
                solution found. Otherwise, indicates to user that no solution could
                be found for points indicated.
"""
def AStar(start, end, maze, points):

    #Calculate heuristic for all points within grid
    for i in range(height):
        for j in range(width):
            points[i][j].h = hypot((points[i][j].x - end.x), (points[i][j].y - end.y))

    frontier = PriorityQueue()

    #Updates values of start point in relation to grid of points
    #Adds start point to the frontier
    points[start.y][start.x].f = points[start.y][start.x].value + points[start.y][start.x].h
    points[start.y][start.x].visited = float("inf")
    frontier.put((points[start.y][start.x].f, points[start.y][start.x].name))

    while frontier.empty() == False:
        value, currName = frontier.get()

        #Determine which point is being referenced based on name obtained
        x = currName % width
        y = currName // width
        currPoint = points[y][x]

        #Checks if the visited value of the point is to be updated
        if value >= points[currPoint.y][currPoint.x].visited:
            continue
        else:
            points[currPoint.y][currPoint.x].visited = value

        #Update visited cell in image to visualize A* algorithm
        maze[currPoint.y][currPoint.x] = colorTraversal(points[currPoint.y][currPoint.x].value, 5000)

        #Checks if current point from frontier is the end point
        if currPoint == end:
            break

        #Get neighboring cells of point
        neighbors = getNeighbors(currPoint)

        for cell in neighbors:

            #Checks if neighboring cell can be used in path solution calculation
            if (checkValidCell(cell) == True and points[cell.y][cell.x].visited == False):

                #Updates value of neighboring cell in relation to grid of points
                #Adds neighboring cell to frontier
                points[cell.y][cell.x].value = points[currPoint.y][currPoint.x].value + 1
                points[cell.y][cell.x].f = points[cell.y][cell.x].value + points[cell.y][cell.x].h
                points[cell.y][cell.x].visited = float("inf")
                points[cell.y][cell.x].parent = points[currPoint.y][currPoint.x]
                frontier.put((points[cell.y][cell.x].f, points[cell.y][cell.x].name))

    #Checks if end point specified by user was visited during search
    if points[end.y][end.x].visited > 0:
        displaySolution(points[start.y][start.x], points[end.y][end.x], imgcopy, points)
    else:
        print("Path Not Found")


"""
Function:       displaySolution
Description:    This function displays a solution for the maze based on the start
                point and end point indicated by the user. This function will only
                execute if a solution is determined
Input:          start   -   The start point indicated by the user within the maze
                end     -   The end point indicated by the user within the maze
                maze    -   The image containing the maze provided by the user
                points  -   A grid of points representing every cell within the image
Result:         Displays a solution path within the maze image based on the start
                point and end point specified by the user
"""
def displaySolution(start, end, maze, points):

    global solutionFound

    solutionFound = True
    solution = deque()
    tempPoint = end

    while tempPoint != start:

        #Adds point to solution deque as long as it is not the start point
        solution.appendleft(tempPoint)

        #Assigns tempPoint as the parent of the previous node added to solution
        tempPoint = points[tempPoint.y][tempPoint.x].parent

    #Adds start point to solution
    solution.appendleft(tempPoint)

    #Loops through every point within the solution path
    for cell in solution:
        maze[cell.y][cell.x] = [0, 0, 255]

    print("Path Found")


#Introduction messages
print("Welcome to the Maze Solver!")
print("Press the ESC key while viewing the image to close the program")
print()
print("Please select a start point and an end point within the maze")
print("Use the left mouse button to plot your points")
print()

#Read image file given by user
imgName = sys.argv[1]

#Convert image file into a binary image
img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
imgcopy = img.copy()
height, width = img.shape[0:2]

#Create thread to run the function of displaying image continuously
t = threading.Thread(target=disp, args=("Maze Solver",img,))
t.start()

#Keep running program until user selects start point and end point
while numClicks < 2:
    if windowOpen == False:
        exit()
    pass

#Draw additional border line messages
print()
print("At this time, you may draw any additional borders onto the maze")
print("Press and hold the left mouse button to indicate the start of the border")
print("Drag your mouse while holding down to indicate the end of the border")
print("Once you have reached the end of your desired border, release the left mouse button")
print("When you are finished, press the SPACE key while viewing the image to continue")
print()

#Keep running program until user finishes drawing additional borders
while drawMode == True:
    if windowOpen == False:
        exit()
    pass

#Ask user which method to use when solving maze
print("Finally, please select which method to use in order to solve the maze")
print("Enter '1' for Breadth First Search (BFS)")
print("Enter '2' for Depth First Search (DFS)")
print("Enter '3' for A Star (A*)")
while True:
    method = input("Method:")
    if (method == "1" or method == "2" or method == "3"):
        print()
        break
    else:
        print()
        print("Invalid value entered. Please enter valid value.")

#Create grid of points based on image dimensions
grid = [[Point(j, i) for j in range(width)] for i in range(height)]

#Perform search based on method entered
if method == "1":
    BFS(startPoint, endPoint, img, grid)
elif method == "2":
    DFS(startPoint, endPoint, img, grid)
else:
    AStar(startPoint, endPoint, img, grid)

#Displays maze with solution if one exists
if solutionFound == True:
    x = threading.Thread(target=disp, args=("Maze Solution",imgcopy,))
    x.start()