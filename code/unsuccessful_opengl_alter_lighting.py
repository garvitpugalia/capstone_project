import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2
from PIL import Image
import pyrr

# Vertex shader
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoords;
layout(location = 2) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 lightPos;

out vec2 fragTexCoords;
out vec3 fragNormal;
out vec3 fragPos;
out vec3 fragLightDir;

void main()
{
    fragTexCoords = texCoords;
    fragNormal = mat3(transpose(inverse(model))) * normal;
    fragPos = vec3(model * vec4(position, 1.0));
    fragLightDir = lightPos - fragPos;
    gl_Position = projection * view * vec4(fragPos, 1.0);
}
"""

# Fragment shader
fragment_shader = """
#version 330 core
in vec2 fragTexCoords;
in vec3 fragNormal;
in vec3 fragPos;
in vec3 fragLightDir;

uniform sampler2D textureSampler;

out vec4 outColor;

void main()
{
    vec3 color = texture(textureSampler, fragTexCoords).rgb;
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(fragLightDir);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * color;
    vec3 ambient = 0.1 * color;
    outColor = vec4(ambient + diffuse, 1.0);
}
"""

def create_shader_program():
    program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return program

def load_image_and_depth(image_path, depth_map_path):
    image = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_map = depth_map / 255.0
    return image, depth_map

def create_3d_model(image, depth_map):
    h, w = depth_map.shape
    vertices = []
    tex_coords = []
    normals = []

    for i in range(h):
        for j in range(w):
            x, y, z = j, i, depth_map[i, j] * 10
            vertices.append((x, y, z))
            tex_coords.append((j / (w - 1), i / (h - 1)))
            normals.append((0, 0, 1))
    
    vertices = np.array(vertices, dtype=np.float32)
    tex_coords = np.array(tex_coords, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    
    return vertices, tex_coords, normals

def main(image_path, depth_map_path, light_position):
    if not glfw.init():
        return
    
    # Fix for Windows 10 + WSL + OpenGL issue
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

    window = glfw.create_window(800, 600, "3D Model with Shadows", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    shader_program = create_shader_program()
    glUseProgram(shader_program)

    image, depth_map = load_image_and_depth(image_path, depth_map_path)
    vertices, tex_coords, normals = create_3d_model(image, depth_map)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(3)

    # Bind vertices
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    # Bind texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    # Bind normals
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(2)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, image)
    glGenerateMipmap(GL_TEXTURE_2D)

    glEnable(GL_DEPTH_TEST)

    model = pyrr.matrix44.create_identity(dtype=np.float32)
    view = pyrr.matrix44.create_look_at(eye=[128, 128, 300], target=[128, 128, 0], up=[0, 1, 0], dtype=np.float32)
    projection = pyrr.matrix44.create_perspective_projection(fovy=45, aspect=800/600, near=0.1, far=1000, dtype=np.float32)
    
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader_program)

        model_loc = glGetUniformLocation(shader_program, "model")
        view_loc = glGetUniformLocation(shader_program, "view")
        projection_loc = glGetUniformLocation(shader_program, "projection")
        light_pos_loc = glGetUniformLocation(shader_program, "lightPos")

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
        glUniform3fv(light_pos_loc, 1, light_position)

        glBindVertexArray(vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glDrawArrays(GL_POINTS, 0, len(vertices))

        glfw.swap_buffers(window)

    # Save the final rendered image with shadows
    width, height = glfw.get_framebuffer_size(window)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output_with_shadows.png")

    glfw.terminate()

if __name__ == "__main__":
    main('/home/garvitpugalia/capstone/data/shadow_altered/19_0.jpg', '/home/garvitpugalia/capstone/data/depth_maps/19_0.jpg', [0, 0, 0])