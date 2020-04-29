import os, tqdm, multiprocessing, platform
from collections import OrderedDict


class ParamShuffler:

    RESULT_PROPERTY_NAME = 'result'
    FINISHED_MSG = 'finished'
    PLATFORM = platform.system()

    def __init__(self, method):
        self.results = []
        self.params = []
        self.method = method

    def run(self, data_dict: OrderedDict):

        # Prepare parameter combinations.
        self.__get_params(data_dict)

        # Get number of CPU and proper imap chunk size.
        cpu_count = multiprocessing.cpu_count()
        chunk_size = int(len(self.params)/cpu_count)
        chunk_size = max(min(chunk_size, 10), 1)

        # Initiate pool parallel processing.
        calculations = []
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for x in tqdm.tqdm(pool.imap(self.wrapper, self.params, chunksize=chunk_size), total=len(self.params)):
            calculations.append(x)

        # Prepare results list.
        for result, param in zip(calculations, self.params):
            new_entry = {}
            for index, item in enumerate(data_dict.items()):
                new_entry[item[0]] = param[index]
            new_entry[self.RESULT_PROPERTY_NAME] = result
            self.results.append(new_entry)

        # Alert developer of end of processing.
        if self.PLATFORM == 'Darwin':
            os.system(f'say "{self.FINISHED_MSG}"')
        elif self.PLATFORM == 'Windows':
            import winsound
            winsound.Beep(1000, 500)
        elif self.PLATFORM == 'Linux':
            os.system('play -nq -t alsa synth {} sine {}'.format(500, 1000))

        return self.results

    def __get_params(self, data_dict: OrderedDict, parameter_values=None, depth=0):

        if parameter_values is None:
            parameter_values = [[k,None] for k, v in data_dict.items()]

        parameters = tuple([x[1] for x in parameter_values])
        if depth == len(parameter_values):
            self.params.append(parameters)
            return

        depth_iteration = list(data_dict.items())[depth][1]
        for x in depth_iteration:
            parameter_values[depth][1] = x
            self.__get_params(data_dict, parameter_values, depth+1)

    def wrapper(self, args):
        return self.method(*args)


if __name__ == "__main__":

    def test_function(a, b):
        return a * b

    ps = ParamShuffler(test_function)

    results = ps.run(
        OrderedDict({
            'a': range(1, 4),
            'b': [15, 27, 42],
        })
    )

    print()
    print('Result type is: ', type(results))
    print('Result content type is: ', type(results[0]))
    [print(x) for x in results]
