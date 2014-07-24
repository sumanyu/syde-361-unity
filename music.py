import pygame
import time

def main():
    while True:
        playMusic()

        while pygame.mixer.music.get_busy():
            for i in range(0, 11):
                adjustVol(0.1 * i)
                time.sleep(5)

def playMusic():
    pygame.init()
    pygame.mixer.music.load("enya.mp3")
    pygame.mixer.music.play(0)

def adjustVol(vol):
    pygame.mixer.music.set_volume(vol)

if __name__ == "__main__":
    main()