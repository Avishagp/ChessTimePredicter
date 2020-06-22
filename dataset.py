import torch
from torch.utils.data import Dataset
from data import data_parser

# Returns the class for the given amount of seconds played.
labels = {'0-3': 0, '4-7': 1, '8 - 12': 2, '13 - 17': 3, '18 - 23': 4, '24 - 30': 5,
          '31 - 40': 6, '41 - 50': 7, '51 - 60': 8, '61 - 120': 9, '121 - 180': 10,
          '181 - 240': 11, '241 - 300': 12, '301 - 420': 13, '421 - 540': 14, '541 - inf': 15}


def seconds_to_class(seconds):
    if seconds <= 3:
        return 0
    elif seconds <= 7:
        return 1
    elif seconds <= 12:
        return 2
    elif seconds <= 17:
        return 3
    elif seconds <= 23:
        return 4
    elif seconds <= 30:
        return 5
    elif seconds <= 40:
        return 6
    elif seconds <= 50:
        return 7
    elif seconds <= 60:
        return 8
    elif seconds <= 120:
        return 9
    elif seconds <= 180:
        return 10
    elif seconds <= 240:
        return 11
    elif seconds <= 300:
        return 12
    elif seconds <= 420:
        return 13
    elif seconds <= 540:
        return 14
    else:
        return 15


class ChessDataset(Dataset):
    def __init__(self, openings, elo_dif, higher_ranker, utc_time, turns):
        self.opening_type = {}

        self.data = {}  # {ID(1,2,3...), [opening_type, elo_difference, am i the higher ranker (0, 1), time, turn number]}
        self.id_to_label = {}  # {ID, turn time}
        self.amount_of_classes = 16

        turn_id = 0
        opening_id = 0

        for key, val in turns.items():
            # Iterate on all turns. Add each turn as a matrix row to data.
            for i, turn in enumerate(val):
                self.data[turn_id] = []
                game_id = int(''.join(filter(str.isdigit, key)))  # Get the game's ID from the key

                # Add the game's opening to the opening type dict if not in there
                if openings[game_id] not in self.opening_type.keys():
                    self.opening_type[openings[game_id]] = opening_id
                    opening_id += 1

                # Add values to data
                # Add opening type number
                self.data[turn_id].append(self.opening_type[openings[game_id]])
                # Add Elo difference
                self.data[turn_id].append(elo_dif[game_id])

                # Check if the higher ranker is the player doing turns right now
                if higher_ranker[game_id] in key:
                    self.data[turn_id].append(1)
                else:
                    self.data[turn_id].append(0)

                # Add hour of game
                self.data[turn_id].append(utc_time[game_id].hour)
                # Add turn number
                self.data[turn_id].append(i)
                # Compute turn time in seconds
                self.id_to_label[turn_id] = seconds_to_class(turn.seconds)

                turn_id += 1

        self.num_of_samples = turn_id

    def __len__(self):
        """Denotes the total number of samples"""
        return self.num_of_samples

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        data = torch.tensor(self.data[index])

        # Get label
        y = self.id_to_label[index]

        return data, y


def main():
    data_parser.parser(r"data/data.txt")
    data = ChessDataset(data_parser.openings, data_parser.elo_dif, data_parser.higher_ranker, data_parser.utc_time,
                        data_parser.turns)

    avg = 0
    i = 0
    all_turns = 0
    max_t = 0
    min_t = 1000
    for turn in data_parser.turns.values():
        for t in turn:
            if t.seconds > max_t:
                max_t = t.seconds
            if t.seconds < min_t:
                min_t = t.seconds
            avg += t.seconds
            all_turns += 1

    avg /= all_turns
    print(f"min: {min_t}, max: {max_t}, avg: {avg}")


if __name__ == "__main__":
    main()
