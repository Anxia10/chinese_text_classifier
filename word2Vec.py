
class InputData:
    def __init__(self,file_name,min_count):
        self.input_file_name = file_name
        self.get_words(min_count)

    def get_words(self,min_count):
        self.input_file = open(self.input_filename)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)

    def evaluate_pair_count(self,window_size):
        return self.sentence_le
class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=50,
                 window_size=5,
                 iteration=1,
                 initial_lr=0.025,
                 min_count=5):
        self.data = InputData(input_file_name, min_count)


