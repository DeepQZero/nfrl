from PIL import Image, ImageDraw




import numpy as np

data = np.load('week1.npy')
print(data[0:5])


def get_players(play_id='75'):
    offensive = set()
    defensive = set()
    for row in data:
        if row[16] == play_id:
            if row[14] == 'home':
                defensive.add(row[9])
            if row[14] == 'away':
                offensive.add(row[9])
    return offensive, defensive


offensive, defensive = get_players()


player_dictionary = {}
for player in offensive:
    player_dictionary[player] = []
for player in defensive:
    player_dictionary[player] = []
player_dictionary[''] = []

all_players = offensive.union(defensive)  # add football?


def get_positions():
    for row in data:
        if row[16] == '75':
            player = row[9]
            player_dictionary[player].append((round(float(row[1]), 1), round(float(row[2]), 1)))
    return player_dictionary


all_positions = get_positions()
print(all_positions)


img = Image.new('L', (1200, 533), color=150)
draw = ImageDraw.Draw(img)
draw.rectangle((0, 0, 100, 533), fill=120, outline=0)
draw.rectangle((1100, 0, 1200, 533), fill=120, outline=0)

draw.line((900, 0, 900, 533), fill=0, width=2)

for player in defensive:
    positions = all_positions[player]
    color = 0 if player in offensive else 255
    first_pos = positions[0]
    draw.rectangle(((int(first_pos[0] * 10)-10, int(first_pos[1] * 10)-10),
                   (int(first_pos[0] * 10)+10, int(first_pos[1] * 10)+10)), fill=color)
    for i in range(len(positions)-1):
        first_pos = positions[i]
        second_pos = positions[i+1]
        new_first_pos = (int(first_pos[0] * 10), int(first_pos[1] * 10))
        new_second_pos = (int(second_pos[0] * 10), int(second_pos[1] * 10))
        draw.line((new_first_pos, new_second_pos), fill=color, width=5)

img.show()
img.save("the_play.png", "PNG")
