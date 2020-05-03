import json
import os, tqdm, multiprocessing, platform
from datetime import datetime


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
        pool = multiprocessing.Pool(processes=cpu_count)
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

    def save_to_csv(self, results, file_name="param-shuffler", separator=";;"):

        file_name_parts = file_name.split('.')
        if len(file_name_parts) == 2:
            file_name = f"{file_name_parts[0]}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.{file_name_parts[1]}"
        else:
            file_name = f"{file_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        column_names = [k for k, v in results[0].items() if k != self.RESULT_PROPERTY_NAME]
        column_names.append(self.RESULT_PROPERTY_NAME)
        data = [column_names]
        for x in results:
            line = []
            for col in column_names:
                if col == self.RESULT_PROPERTY_NAME: continue
                line.append(json.dumps(x[col]))
            line.append(json.dumps(x[self.RESULT_PROPERTY_NAME]))
            data.append(line)

        with open(file_name, "w") as csv_file:
            for line in data:
                w = separator.join(line) + '\n'
                csv_file.write(w)


def test_function(a, b):
    return a * b


if __name__ == "__main__":

    ps = ParamShuffler(test_function)

    results = ps.run({
        'a': range(1, 4),
        'b': [5, 9, 11]
    })

    ps.save_to_csv(results)

    print('Result type is: ', type(results))
    print('Result content type is: ', type(results[0]))
    [print(x) for x in results]
