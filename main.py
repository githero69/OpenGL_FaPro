import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from TextureLoader import load_texture
import numpy as np
from Camera import Camera
from Component import Component
from Step import Step
from Tool import Tool
from Point import Point
import math
import csv

CONST_SCALE = 1
cam = Camera()
WIDTH, HEIGHT = 1280, 720
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward = False, False, False, False


def to_int(arg):
    return int(float(arg) * CONST_SCALE)


def remove_duplicates(lis):
    newlist = []
    newlist.append(lis[0])
    newlist_count = 0

    for i in range(len(lis)):
        if lis[i].y != newlist[newlist_count].y:
            newlist.append(lis[i])
            newlist_count += 1
    return newlist


def calculate_circle(tool):
    points = []
    radius = tool.diameter / 2

    for i in range(91 * CONST_SCALE):
        pointx = int(radius * math.cos((i / CONST_SCALE) * math.pi / 180) * CONST_SCALE)
        pointy = int(radius * math.sin((i / CONST_SCALE) * math.pi / 180) * CONST_SCALE)

        point = Point(pointx, pointy)
        points.append(point)
    return points


def calculate_height(step, component):
    oldheight = component.getArrayAt(step.x, step.y)
    diff = oldheight - (step.z - 15)
    if diff > 0:
        newheight = oldheight - diff
    else:
        newheight = oldheight
    return newheight


# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False


# do the movement, call this function in the main loop
def do_movement():
    if left:
        cam.process_keyboard("LEFT", 1.0)
    if right:
        cam.process_keyboard("RIGHT", 1.0)
    if forward:
        cam.process_keyboard("FORWARD", 1.0)
    if backward:
        cam.process_keyboard("BACKWARD", 1.0)


# the mouse position callback function
def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    cam.process_mouse_movement(xoffset, yoffset)


vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_offset;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
// uniform mat4 move;
out vec2 v_texture;
void main()
{
    vec3 final_pos = a_position + a_offset;
    gl_Position =  projection * view * model * vec4(final_pos, 1.0f);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330
in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    out_color = texture(s_texture, v_texture);
}
"""


# the window resize callback function
def window_resize_clb(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


x = 70
y = 70
z = 30
steps = []

with open('tasche.csv') as tasche:
    reader = csv.reader(tasche, delimiter=' ')
    for zeile in reader:
        step = Step(to_int(zeile[0]),
                    to_int(zeile[1]),
                    to_int(zeile[2]),
                    to_int(zeile[3]),
                    to_int(zeile[4]),
                    to_int(zeile[5]))
        steps.append(step)

tool = Tool(8, 30, 0, 2)
component = Component(x, y, z)

points = calculate_circle(tool)
new_points = remove_duplicates(points)
new_points.reverse()

# for j in range(len(steps)):
#    height = calculateHeight(steps[j], component)
#    for i in range(newPoints[0].y):
#        component.removeCircle(steps[j].x, steps[j].y, newPoints[i].x, newPoints[i].y, height)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(WIDTH, HEIGHT, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize_clb)
# set the mouse position callback
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window, key_input_clb)
# capture the mouse cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# make the context current
glfw.make_context_current(window)

#              positions        texture_coords
cube_buffer = [-0.5, -0.5, 0.5, 0.0, 0.0,
               0.5, -0.5, 0.5, 1.0, 0.0,
               0.5, 0.5, 0.5, 1.0, 1.0,
               -0.5, 0.5, 0.5, 0.0, 1.0,

               -0.5, -0.5, -0.5, 0.0, 0.0,
               0.5, -0.5, -0.5, 1.0, 0.0,
               0.5, 0.5, -0.5, 1.0, 1.0,
               -0.5, 0.5, -0.5, 0.0, 1.0,

               0.5, -0.5, -0.5, 0.0, 0.0,
               0.5, 0.5, -0.5, 1.0, 0.0,
               0.5, 0.5, 0.5, 1.0, 1.0,
               0.5, -0.5, 0.5, 0.0, 1.0,

               -0.5, 0.5, -0.5, 0.0, 0.0,
               -0.5, -0.5, -0.5, 1.0, 0.0,
               -0.5, -0.5, 0.5, 1.0, 1.0,
               -0.5, 0.5, 0.5, 0.0, 1.0,

               -0.5, -0.5, -0.5, 0.0, 0.0,
               0.5, -0.5, -0.5, 1.0, 0.0,
               0.5, -0.5, 0.5, 1.0, 1.0,
               -0.5, -0.5, 0.5, 0.0, 1.0,

               0.5, 0.5, -0.5, 0.0, 0.0,
               -0.5, 0.5, -0.5, 1.0, 0.0,
               -0.5, 0.5, 0.5, 1.0, 1.0,
               0.5, 0.5, 0.5, 0.0, 1.0]

cube_buffer = np.array(cube_buffer, dtype=np.float32)

cube_indices = [0, 1, 2, 2, 3, 0,
                4, 5, 6, 6, 7, 4,
                8, 9, 10, 10, 11, 8,
                12, 13, 14, 14, 15, 12,
                16, 17, 18, 18, 19, 16,
                20, 21, 22, 22, 23, 20]

cube_indices = np.array(cube_indices, dtype=np.uint32)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# cull
# glEnable(GL_CULL_FACE)
# glCullFace(GL_BACK)
# glFrontFace(GL_CCW)

# VAO, VBO and EBO
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
EBO = glGenBuffers(1)

# cube VAO
glBindVertexArray(VAO)
# cube Vertex Buffer Object
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, cube_buffer.nbytes, cube_buffer, GL_STATIC_DRAW)
# cube Element Buffer Object
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

# cube vertices
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube_buffer.itemsize * 5, ctypes.c_void_p(0))
# cube textures
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, cube_buffer.itemsize * 5, ctypes.c_void_p(12))

textures = glGenTextures(1)
load_texture("crate.jpg", textures)

# instance VBO
instance_array = []
offset = 1

# for z in range(0, 10, 1):
#    for y in range(0, 10, 1):
#        for x in range(0, 10, 1):
#            translation = pyrr.Vector3([0.0, 0.0, 0.0])
#            translation.x = x + offset
#            translation.y = y + offset
#            translation.z = z + offset
#            instance_array.append(translation)
for x in range(0, 70, 1):
    for y in range(0, 70, 1):
        for z in range(0, 30, 1):
            translation = pyrr.Vector3([0.0, 0.0, 0.0])
            translation.x = x + offset
            translation.y = y + offset
            translation.z = z + offset
            instance_array.append(translation)

# instance_array.remove(pyrr.Vector3([1.0, 1.0, 1.0]))
# instance_array.remove(pyrr.Vector3([2.0, 1.0, 1.0]))
len_of_instance_array = len(instance_array)  # do this before you flatten the array
instance_array = np.array(instance_array, np.float32).flatten()
# instance_array[0] = None
# instance_array[207] = None
# instance_array[213] = None
# instance_array[420] = None
# instance_array[3255] = None
# instance_array[6300] = None
# instance_array[157500] = None
# instance_array[157710] = None
# instance_array[156] = None
# instance_array[201] = None
# instance_array[0] = None
# instance_array[90] = None
# instance_array[6300] = None

instanceVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, instanceVBO)
glBufferData(GL_ARRAY_BUFFER, instance_array.nbytes, instance_array, GL_STATIC_DRAW)

glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
glVertexAttribDivisor(2, 1)  # 1 means, every instance will have it's own translate

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 100)
cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([-50.0, -50.0, -100.0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")
move_loc = glGetUniformLocation(shader, "move")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(model_loc, 1, GL_FALSE, cube_pos)

index = 0
# the main application loop
while not glfw.window_should_close(window):
    do_movement()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # move = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, glfw.get_time() * 8]))
    # glUniformMatrix4fv(move_loc, 1, GL_FALSE, move)

    view = cam.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    height = calculate_height(steps[index], component)
    for i in range(new_points[0].y):
        component.remove_circle(steps[index].x, steps[index].y, new_points[i].x, new_points[i].y, height)

    for outer in range(0, x, 1):
        for inner in range(0, y, 1):
            instance_array[inner * 3 + outer * 90] = None
            # instance_array_height = component.array[outer][inner] * 6300
            # instance_array_pos = outer * 90 + inner * 3
            # if instance_array_height <= 441000:
            #    for instance_array_height in range(instance_array_height, 441000, 6300):

    glBufferData(GL_ARRAY_BUFFER, instance_array.nbytes, instance_array, GL_STATIC_DRAW)
    glDrawElementsInstanced(GL_TRIANGLES, len(cube_indices), GL_UNSIGNED_INT, None, len_of_instance_array)

    if index < len(steps) - 1:
        index += 1
    else:
        print("finito")

    glfw.swap_buffers(window)
    glfw.poll_events()

# terminate glfw, free up allocated resources
glfw.terminate()
