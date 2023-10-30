import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torchvision import transforms
from scipy.interpolate import make_interp_spline

def add_stroke(image, parameters):
    points, stroke_widths, stroke_colors = parameters[:8], parameters[8:12], parameters[12:]

    # Convert points to numpy array
    points = np.array(points).reshape((4, 2))

    for i in range(len(points)):
        points[i][0] *= image.size[0]
        points[i][1] *= image.size[1]

    num_points = int(calc_dist(points) / 2)
    stroke_colors = np.array(stroke_colors).reshape((4, 4)).T

    # Interpolate using NURBS
    x, y = points[:, 0], points[:, 1]
    t = range(len(points))
    spl = make_interp_spline(t, np.c_[x, y], k=3)
    path = spl(np.linspace(0, points.shape[0] - 1, num_points))

    # Draw the stroke on the image
    stroke_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke_image)

    for i in range(1, len(path)):
        pt1 = tuple(map(int, path[i - 1]))
        pt2 = tuple(map(int, path[i]))
        # Interpolate stroke width
        t = i / len(path)  # Interpolation parameter
        interpolated_width = int(np.interp(t, [i / (len(stroke_widths) - 1) for i in range(len(stroke_widths))], stroke_widths))
        interpolated_color = tuple([int(np.interp(t, [i / (len(stroke_widths) - 1) for i in range(len(stroke_widths))], stroke_colors[j])) for j in range(4)])
        draw.ellipse([pt2[0] - interpolated_width, pt2[1] - interpolated_width,
                      pt2[0] + interpolated_width, pt2[1] + interpolated_width], fill=interpolated_color)

    # Blend the stroke image with the original image using alpha blending
    blended_image = Image.alpha_composite(image.convert("RGBA"), stroke_image)

    return blended_image

def calc_dist(points):
    total_dist = 0.0
    for i in range(1, len(points)):
        x_dist = points[i][0] - points[i-1][0]
        y_dist = points[i][1] - points[i-1][1]
        c_dist = np.sqrt(x_dist * x_dist + y_dist * y_dist)
        total_dist += c_dist
    return total_dist

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Box(0, 1, (28,))
        self.observation_space = gym.spaces.Box(0, 255, (2, 4, 512, 512))

        self.transform = transforms.ToTensor()
        self.goal_image = Image.open('cat.png').convert('RGBA').resize((512, 512))
        self.im_size = self.goal_image.size
        self.goal_image = self.transform(self.goal_image).unsqueeze(dim=0)
        self.new_image = Image.new("RGBA", self.im_size, (0, 0, 0, 0))

        self.criterion = nn.MSELoss()
        self.step_num = 0
        
    def reset(self, seed=42):
        # Reset the environment and return initial state
        self.new_image = Image.new("RGBA", self.im_size, (0, 0, 0, 0))
        self.step_num = 0
        return torch.cat([self.transform(self.new_image).unsqueeze(dim=0), self.goal_image], dim=0)

    def step(self, action):
        self.new_image = add_stroke(self.new_image, action)
        new_image_tensor = self.transform(self.new_image).unsqueeze(dim=0)
        next_state = torch.cat([new_image_tensor, self.goal_image], dim=0)
        loss = self.criterion(new_image_tensor, self.goal_image)
        self.step_num += 1
        # Perform action and return next state, reward, done, and additional info
        return next_state, -loss.item(), self.step_num > 100, False, {}

    def render(self, mode='human'):
        # Render the environment for visualization
        pass