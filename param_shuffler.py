import os, tqdm, multiprocessing, platform


class ParamShuffler:

    RESULT_PROPERTY_NAME = 'result'
    FINISHED_MSG = 'finished'
    MAX_CHUNK_SIZE = 10
    PLATFORM = platform.system()

    def __init__(self, method):
        self.results = []
        self.params = []
        self.method = method

    def run(self, data_dict):

        # Prepare parameter combinations.
        self.__build_params_list(data_dict)

        # Get number of CPU and proper imap chunk size.
        cpu_count = multiprocessing.cpu_count()
        chunk_size = int(len(self.params)/cpu_count)
        chunk_size = max(min(chunk_size, self.MAX_CHUNK_SIZE), 1)

        # Initiate pool parallel processing.
        calculations = []
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for x in tqdm.tqdm(pool.imap(self.wrapper, self.params, chunksize=chunk_size), total=len(self.params)):
            calculations.append(x)

        # Prepare results list.
        for result, param in zip(calculations, self.params):
            param[self.RESULT_PROPERTY_NAME] = result
            self.results.append(param)

        # Alert developer of end of processing and return results.
        self.__alert_dev()
        return self.results

    def __alert_dev(self):
        if self.PLATFORM == 'Darwin':
            os.system(f'say "{self.FINISHED_MSG}"')
        elif self.PLATFORM == 'Windows':
            import winsound
            winsound.Beep(1000, 500)
        elif self.PLATFORM == 'Linux':
            os.system('play -nq -t alsa synth {} sine {}'.format(500, 1000))

    def __build_params_list(self, data_dict, new_entry=None, keys_to_check=None, depth=0):
        if new_entry is None:
            new_entry = {k:None for k, v in data_dict.items()}
            keys_to_check = [k for k, v in data_dict.items()]

        if depth == len(data_dict.items()):
            self.params.append(new_entry)
            return

        depth_iteration = keys_to_check[depth]
        for x in data_dict[depth_iteration]:
            new_entry[depth_iteration] = x
            self.__build_params_list(data_dict, new_entry.copy(), keys_to_check, depth+1)

    def wrapper(self, args):
        return self.method(**args)


if __name__ == "__main__":

    def test_function(a, b):
        return a * b

    ps = ParamShuffler(test_function)

    results = ps.run({
        'a': range(1, 4),
        'b': [5, 9, 11]
    })

    print()
    print('Result type is: ', type(results))
    print('Result content type is: ', type(results[0]))
    [print(x) for x in results]
