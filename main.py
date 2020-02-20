from tkinter import *
import random
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt
import time
from threading import Thread
import pygame
from pygame.locals import *

    
pygame.init()

def aide():
    showinfo("alerte", "test")

# fenetre Lr
def Lrfenetre():
    Lrfen = Toplevel()
    Lrfen.configure(bg="black")
    Lrfen.title("Lr")
    sLr = Spinbox(Lrfen, from_=0, to=1)
    sLr.pack()

# fenetre y
def yfenetre():
    yfen = Toplevel()
    yfen.configure(bg="black")
    yfen.title("y")
    sy = Spinbox(yfen, from_=0, to=1)
    sy.pack()
    
# fenetre num_episode
def num_episodefenetre():
    num_episodefen = Toplevel()
    num_episodefen.configure(bg="black")
    num_episodefen.title("num_episode")
    snum = Spinbox(num_episodefen, from_=0, to=100000, textvariable=float)
    snum.pack()




    

class Game:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3

    ACTIONS = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP]

    ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT"]

    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0)
    }

    num_actions = len(ACTIONS)

    def __init__(self, n, m, wrong_action_p=0.1, alea=False):
        self.n = n
        self.m = m
        self.wrong_action_p = wrong_action_p
        self.alea = alea
        self.generate_game()



    def _position_to_id(self, x, y):
        """Donne l'identifiant de la position entre 0 et 15"""
        return x + y * self.n

    def _id_to_position(self, id):
        """Réciproque de la fonction précédente"""
        return (id % self.n, id // self.n)

    def generate_game(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        hole = random.choice(cases)
        cases.remove(hole)
        start = random.choice(cases)
        cases.remove(start)
        end = random.choice(cases)
        cases.remove(end)
        block = random.choice(cases)
        cases.remove(block)

        self.position = start
        self.end = end
        self.hole = hole
        self.block = block
        self.counter = 0
        
        if not self.alea:
            self.start = start
        return self._get_state()
    
    def reset(self):
        son = pygame.mixer.Sound("couin.wav")
        son.play()
        if not self.alea:
            self.position = self.start
            self.counter = 0
            return self._get_state()
        else:
            return self.generate_game()
        


    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille

    def _get_state(self):
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.hole, self.block]]
        return self._position_to_id(*self.position)

    def move(self, action):
        """
        takes an action parameter
        :param action : the id of an action
        :return ((state_id, end, hole, block), reward, is_final, actions)
        """
        
        self.counter += 1

        if action not in self.ACTIONS:
            raise Exception("Invalid action")

        # random actions sometimes (2 times over 10 default)
        choice = random.random()
        if choice < self.wrong_action_p:
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        if self.block == (new_x, new_y):
            return self._get_state(), -1, False, self.ACTIONS
        elif self.hole == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), -10, True, None
        elif self.end == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), 10, True, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.ACTIONS
        elif self.counter > 190:
            self.position = new_x, new_y
            return self._get_state(), -10, True, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), -1, False, self.ACTIONS

# environnement
    def print(self):
        str = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.position:
                    str += "x"
                elif (i, j) == self.block:
                    str += "¤"
                elif (i, j) == self.hole:
                    str += "o"
                elif (i, j) == self.end:
                    str += "@"
                else:
                    str += "."
            str += "\n"
        print(str)

def RL():
    states_n = 100
    actions_n = 4
    Q = np.zeros([states_n, actions_n])

    # Set learning parameters
    lr = .85
    y = .99
    num_episodes = 1000
    cumul_reward_list = []
    actions_list = []
    states_list = []
    game = Game(10, 10, 0) 
    for i in range(num_episodes):
        actions = []
        s = game.reset()
        states = [s]
        cumul_reward = 0
        d = False
        while True:
            # on choisit une action aléatoire avec une certaine probabilité, qui décroit
          
            Q2 = Q[s,:] + np.random.randn(1, actions_n)*(1. / (i +1))
            a = np.argmax(Q2)
            s1, reward, d, _ = game.move(a)
            Q[s, a] = Q[s, a] + lr*(reward + y * np.max(Q[s1,:]) - Q[s, a]) # Fonction de mise à jour de la Q-table
            cumul_reward += reward
            s = s1
            actions.append(a)
            states.append(s)
            game.print()
            if d == True:
                break
            time.sleep(0.01)
        states_list.append(states)
        actions_list.append(actions)
        cumul_reward_list.append(cumul_reward)
        
    print("Score over time: " +  str(sum(cumul_reward_list[-100:])/100.0))

    game.reset()
    game.print()

    son2 = pygame.mixer.Sound("mario.wav")
    son2.play()

    messagebox.showinfo("information","Score over time: " +  str(sum(cumul_reward_list[-100:])/100.0))

    
   


    plt.plot(cumul_reward_list[:100])
    plt.ylabel('Cumulative reward')
    plt.xlabel('Étape')
    plt.show()





   

# création de la fenetre main
fenetre = Tk()
fenetre.title("Qlearning")


#barre menu
   
menubar = Menu(fenetre)

menu1 = Menu(menubar, tearoff=0)
menu1.add_command(label="Lr", command=Lrfenetre)
menu1.add_command(label="y", command=yfenetre)
menu1.add_command(label="num_episode", command=num_episodefenetre)
menubar.add_cascade(label="Editer", menu=menu1)

menu2 = Menu(menubar, tearoff=0)
menu2.add_command(label="A propos", command=aide)
menubar.add_cascade(label="Aide", menu=menu2)

menue3 =Menu(menubar, tearoff=0)
menue3.add_command(label="quitter", command=quit)
menubar.add_cascade(label="Option", menu=menue3)

fenetre.config(menu=menubar)

# frame 1
Frame1 = Frame(fenetre, borderwidth=2, relief=GROOVE)
Frame1.pack(side=LEFT, padx=30, pady=30)

# frame 2
Frame2 = Frame(fenetre, borderwidth=2, relief=GROOVE)
Frame2.pack(side=RIGHT, padx=30, pady=30)

#label1
champ_label = Label(Frame1, text="VERSION 3")
champ_label.pack()

# bouton start
bouton_start = Button(Frame1, text="start", command=RL)
bouton_start.pack()


# Lr

# y

# num_episode


# on tient la fenetre ouverte
fenetre.mainloop()


