import pygame
import numpy as np
from scipy.integrate import ode
import random
import sys
from itertools import combinations
import math
#SIMULATION GLOBALS
G = 6.674e-11 # N kg-2 m^2
ACCELERATOR = 0.1
cur_time = 0.
dt = 0.033

#PYGAME GLOBALS
BLACK = (0, 0, 0)
COLORS = [(252, 186, 3), (255, 0, 0), (0, 255, 0), (0, 0, 255), (181, 3, 252), (3, 190, 252), (75, 113, 125)]
win_width = 1280
win_height = 640

import random

def random_location():
    while True:
        first_value = random.randint(-640, 640)
        second_value = random.randint(-320, 320)
        
        if not (-50 <= first_value <= 50 and -50 <= second_value <= 50):
            return [first_value, second_value, 0, 0]

def random_one_to_ten():
    return random.randint(3, 10)

def pick_random_color():
    return random.choice(COLORS)

def to_screen(x, y, win_width, win_height):
    return win_width//2 + x, win_height//2 - y

def from_screen(x, y, win_width, win_height):
    return x - win_width//2, win_height//2 - y

def mix_color(Color1, Color2):
    result_color = (Color1[0] + Color2[0], Color1[1] + Color2[1], Color1[2] + Color2[2])
    return tuple(min(255, max(0, c)) for c in result_color)

def load_image(name):
    image = pygame.image.load(name)
    return image

def circle_collision(A, B):
    x1 = A.state[0]
    y1 = A.state[1]
    r1 = A.radius

    x2 = B.state[0]
    y2 = B.state[1]
    r2 = B.radius

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance <= (r1 + r2)
    
def handle_collision(A, B, group_body):
    C = CelestialBody()
    C.state[:2] = A.state[:2]
    
    C.color = mix_color(A.color, B.color)

    if A in group_body:
        group_body.remove(A)
    if B in group_body:
        group_body.remove(B)
    
    return C

class Spaceship(pygame.sprite.Sprite):
    def __init__(self, imagefile):
        pygame.sprite.Sprite.__init__(self)
        self.image = load_image(imagefile)
        self.rect = self.image.get_rect()
        self.mass = 1
        self.radius = 3
        self.state = np.array([0., 0., 0., 0.]) #x, y, vx, vy
        
        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_initial_value(self.state, cur_time)

    def set_image(self, imagefile):
        self.image = load_image(imagefile)
        
    def f(self, t, state):
        x = self.state[0]
        y = self.state[1]

        vx = self.state[2]
        vy = self.state[3]

        return [vx, vy, 0, 0]

    def rotate(self, angle):
        self.image.rot = pygame.transform.rotate(self.image, angle)

    def set_pos(self, pos):
        self.rect.x = pos[0] - self.rect.width//2
        self.rect.y = pos[1] - self.rect.height//2
    
    def step(self, group_body):
        self.solver.integrate(cur_time + dt)
        y = self.solver.y
        acceleration = np.array([0.0, 0.0])  # Initialize acceleration

        for body in group_body:
            gravitational_force = body.pull(self)
            acceleration += gravitational_force / self.mass

        y[2:4] += acceleration * dt
        self.state = y

    def draw(self, screen):
        self.set_pos(to_screen(self.state[0], self.state[1], win_width, win_height))
        screen.blit(self.image, self.rect)
    
    def isColliding(self, other):
        x1 = self.state[0]
        y1 = self.state[1]
        r1 = self.radius

        x2 = other.state[0]
        y2 = other.state[1]
        r2 = other.radius

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance <= (r1 + r2)

    def handleCollision(self, screen):
        self.set_image("/Project/assets/explosion.png")
        self.update()
        self.draw(screen)    

class CelestialBody(pygame.sprite.Sprite):
    def __init__(self, state):
        pygame.sprite.Sprite.__init__(self)
        pos = random_location()
        self.radius = random_one_to_ten()
        self.image = pygame.Surface([self.radius*4, self.radius*4])
        self.rect = self.image.get_rect()
        self.color = pick_random_color()
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius, self.radius)
        self.mass = self.radius * 10000000000
        self.state = state
        self.state = np.array(self.state)
        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_initial_value(self.state, cur_time)

    def set_state(self, state):
        self.state = state

    def f(self, t, state):
        x = self.state[0]
        y = self.state[1]

        vx = self.state[2]
        vy = self.state[3]

        return [vx, vy, 0, 0]
    
    def pull(self, other):
        other_pos = other.state[:2]
        distance = self.state[:2] - other_pos
        distance_norm = np.linalg.norm(distance)
        unit_vector = distance / distance_norm
        F = (G * self.mass * other.mass) / (distance_norm * distance_norm)
        return F * unit_vector

    def step(self, group_body):
        self.solver.integrate(cur_time + dt)
        y = self.solver.y
        acceleration = np.array([0., 0.])

        for body in group_body:
            if body != self:
                gravitational_force = body.pull(self)
                acceleration += gravitational_force / self.mass

        y[2:4] += acceleration * dt
        self.state = y

    def set_pos(self, pos):
        self.rect.x = pos[0] - self.rect.width//2
        self.rect.y = pos[1] - self.rect.height//2

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, to_screen(self.state[0], self.state[1], win_width, win_height), self.radius, self.radius)

    
class Ring(pygame.sprite.Sprite):
    def __init__(self, body):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([body.radius, body.radius])
        self.rect = self.image.get_rect()
        self.body = body
        self.span = body.radius * 24
        pygame.draw.circle(self.image, self.body.color, (self.span, self.span), self.span, self.span)
    
    def set_pos(self, pos):
        self.rect.x = pos[0] - self.rect.width//2
        self.rect.y = pos[1] - self.rect.height//2

    def draw(self, screen):
        pygame.draw.circle(screen, self.body.color, to_screen(self.body.state[0], self.body.state[1], win_width, win_height), self.span , 1)
    
    def pull(self, other):
        other_pos = other.state[:2]
        distance = self.body.state[:2] - other_pos
        distance_norm = np.linalg.norm(distance)
        if distance_norm <= 0.01:
            other.state[2:4] = 0
            return 0
        elif distance_norm <= self.span:
            unit_vector = distance / distance_norm
            F = (G * self.body.mass * other.mass) / (distance_norm*distance_norm)
            return F * unit_vector
        return 0 
    
    

def main():
    pygame.init()
    paused = False
    win_width = 1280
    win_height = 640
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('Gravitational Simulation')

    spaceship = Spaceship('Project/assets/up.png')
    group_body = []
    group_ring = []

    # Number of celestial bodies and rings you want to create
    num_bodies = 10
    for _ in range(num_bodies):
        pos = random_location()
        body = CelestialBody(state = pos)
        group_body.append(body)

        # ring = Ring(body)
        # group_ring.append(ring)

    spaceship.set_pos(to_screen(0, 0, win_width, win_height))

    print("Press q to quit.")
    print("Press p to pause.")
    print("Press r to resume.")
    print("Arrows to move.")

    while True:
        event = pygame.event.poll()
        keys = pygame.key.get_pressed()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
             paused = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
             paused = True
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT] and not paused:
                spaceship.set_image("Project/assets/up_right.png")
                spaceship.state[2] += ACCELERATOR
                spaceship.state[3] += ACCELERATOR
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT] and not paused:
                spaceship.set_image("Project/assets/up_left.png")
                spaceship.state[2] -= ACCELERATOR
                spaceship.state[3] += ACCELERATOR
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT] and not paused:
                spaceship.set_image("Project/assets/down_left.png")
                spaceship.state[2] -= ACCELERATOR
                spaceship.state[3] -= ACCELERATOR
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT] and not paused:
                spaceship.set_image("Project/assets/down_right.png")
                spaceship.state[2] += ACCELERATOR
                spaceship.state[3] -= ACCELERATOR
        elif keys[pygame.K_LEFT] and not paused:
                spaceship.state[2] -= ACCELERATOR
                spaceship.set_image("Project/assets/left.png")
        elif keys[pygame.K_RIGHT] and not paused:
                spaceship.state[2] += ACCELERATOR
                spaceship.set_image("Project/assets/right.png")
        elif keys[pygame.K_DOWN] and not paused:
                spaceship.state[3] -= ACCELERATOR
                spaceship.set_image("Project/assets/down.png")
        elif keys[pygame.K_UP] and not paused:
                spaceship.state[3] += ACCELERATOR
                spaceship.set_image("Project/assets/up.png")
                
        else:
            pass

        if not paused:
            pygame.display.flip()
            screen.fill(BLACK)
            for A, B in combinations(group_body, 2):
                if circle_collision(A, B):
                    new_state = (A.state + B.state) / 2
                    new_state[2:4] = 0.
                    C = CelestialBody(state = new_state)
                    C.color = mix_color(A.color, B.color)
                    C.radius = A.radius + B.radius
                    if A in group_body:
                        group_body.remove(A)
                    if B in group_body:
                            group_body.remove(B)
                    group_body.append(C)

                spaceship.step(group_body)
                for body in group_body:
                    body.draw(screen)
                    body.step(group_body)
                    if spaceship.isColliding(body):
                        spaceship.handleCollision(screen)
                        pygame.display.flip()
                        paused = True
                        

                # for ring in group_ring:
                #     ring.draw(screen)
                spaceship.update()
                spaceship.draw(screen)

                global cur_time
                cur_time += dt

if __name__ == '__main__':
    main()