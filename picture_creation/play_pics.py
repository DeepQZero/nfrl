from PIL import Image, ImageDraw
import pickle
from multiprocessing import Pool
import os


class PlayVisualizer:
    """Creates .png graphic of game play for first 3 seconds."""
    def __init__(self, play_file: str, granularity: int) -> None:
        """Initialize fields."""
        self.game_id = None
        self.play_id = None
        self.snap_frame = None
        self.los = None
        self.player_info = None
        self.play_direction = None
        self.off_players = None
        self.def_players = None
        self.gran = granularity
        self.init_everything(play_file)

    def init_everything(self, play_file: str) -> None:
        """Fills fields with values."""
        infile = '../data/dictionaries/plays/' + play_file
        play_dict = pickle.load(open(infile, "rb"))
        self.game_id = play_dict['game_id']
        self.play_id = play_dict['play_id']
        self.snap_frame = play_dict['snap_frame']
        self.los = play_dict['los']
        self.player_info = play_dict['player_info']
        self.play_direction = play_dict['play_direction']
        self.off_players = play_dict['off_players']
        self.def_players = play_dict['def_players']

    def det_bounds(self, team: str) -> tuple:
        """Determines the direction to crop the image."""
        direction, los = self.play_direction, self.los
        if direction == 'left' and team == 'def':
            return los-8, los+22
        elif direction == 'right' and team == 'def':
            return los+18, los+48
        elif direction == 'left' and team == 'off':
            return los, los+30
        elif direction == 'right' and team == 'off':
            return los+10, los+40
        else:
            raise Exception('Incorrect direction or side')

    def plot_all(self, team: str) -> None:
        """Plots plays of specified teams and saves pictures."""
        if team == 'def':
            self.plot_pic('def')
        elif team == 'off':
            self.plot_pic('off')
        elif team == 'all':
            self.plot_pic('def')
            self.plot_pic('off')
        else:
            pass

    def plot_pic(self, team: str) -> None:
        """Plots a single play for specified team and saves picture."""
        img = Image.new('L', (self.gran * 160, self.gran * 54), color=0)
        draw = ImageDraw.Draw(img)
        platoon = self.off_players if team == 'off' else self.def_players
        for player_id in platoon:
            self.draw_player_route(player_id, draw)
        new_img = self.crop_flip_image(team, img)
        filename = '../data/play_pics/' + team + '/class_1/' + \
                   self.game_id + "-" + self.play_id + '.png'
        new_img.save(filename, 'PNG', quality=100)

    def draw_player_route(self, player_id: str,
                          draw: ImageDraw.ImageDraw) -> None:
        """Draws entire single route for player and start square."""
        route = self.player_info[player_id]
        snap_route = route[self.snap_frame - 1:]
        self.plot_square(snap_route[0], draw)
        for frame in range(min(len(snap_route) - 1, 30)):
            self.draw_route_frame(snap_route, frame, draw)

    def plot_square(self, route_start: dict,
                    draw: ImageDraw.ImageDraw) -> None:
        """Plots square to signify player at start of route."""
        x_pos, y_pos = route_start['x_pos'], route_start['y_pos']
        draw.rectangle((self.gran * (int(x_pos) + self.gran * 10 - 1),
                        self.gran * (int(y_pos) + 1),
                        self.gran * (int(x_pos) + self.gran * 10 + 1),
                        self.gran * (int(y_pos) - 1)), fill=255)

    def draw_route_frame(self, route: list, frame: int,
                         draw: ImageDraw.ImageDraw) -> None:
        """Draws single frame of route for single player."""
        first_pos = (self.gran * int((route[frame]['x_pos']) + 20),
                     self.gran * int(route[frame]['y_pos']))
        second_pos = (self.gran * int((route[frame + 1]['x_pos']) + 20),
                      self.gran * int(route[frame + 1]['y_pos']))
        draw.line((first_pos, second_pos), fill=255, width=self.gran*1)

    def crop_flip_image(self, team: str, image: Image.Image) -> Image.Image:
        """Crops and possibly flips image."""
        left_bound, right_bound = self.det_bounds(team)
        new_img = image.crop((self.gran * left_bound,
                              self.gran * 0,
                              self.gran * right_bound,
                              self.gran * 54))
        if self.play_direction == 'left':
            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
        return new_img


def pooler(play_file: str) -> None:
    "Plots a single play. Used for multiprocessing."
    polisher = PlayVisualizer(play_file, granularity=2)
    polisher.plot_all('all')


if __name__ == "__main__":
    path, dirs, files = next(os.walk("../data/dictionaries/plays"))
    p = Pool(6)
    p.map(pooler, files)
    p.close()
    p.join()
