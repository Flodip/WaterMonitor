import psycopg2
import matplotlib.pyplot as plt
import sys, traceback, os
import numpy as np
import math
import configparser
import time
# from scipy.stats import sem, t
# from scipy import mean
# import itertools
# from scipy import optimize # to estimate sine function from some points

# to check
# filtre pass-bande: butterworth, pass-bas, pass-haut
# cours signaux et systèmes: partie signaux
# dc blocker: applatir signal https://dsp.stackexchange.com/questions/28001/how-can-i-center-an-audio-signal


class SignalProcessing:
    # make additional version that does not need config file
    def __init__(self):
        try:
            # load config file
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")

            self.slope = float(self.config["Signal"]["slope"])  # slope used to detect noises
            self.seg_width = int(self.config["Signal"]["min_period"]) // 2  # width of segments analyzed to detect noise

            delta = float(self.config["Signal"]["delta"])
            if not np.isscalar(delta):
                raise Exception('Input argument delta must be a scalar')
            elif delta <= 0:
                raise Exception('Input argument delta must be positive')
            else:
                self.delta = delta  # value of the noise ignored during peak detect

            self.data = []  # values of signal
            self.timestamps = []  # timestamps of signal
            self.max_tab = []  # max peaks
            self.min_tab = []  # min peaks
            self.noise_tab = []  # noise values
            self.noise_seg_tab = []  # segments of noise, indice i = start, i+1 = end
            self.wave_seg_tab = []  # segments of waves, indice i = start, i+1 = end
            self.wave_seg_objects = []

            # self.noise_det()
            # if self.noise_seg_tab:
            #     self.peak_det()
            # if self.min_tab and self.max_tab:
            #     self.extend_peaks()
            # self.save()

        except (configparser.Error, KeyError):
            print("Invalid config.ini file format")
            sys.exit(1)
        except ValueError:
            print("Invalid parsing from config.ini")
            sys.exit(1)
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

    def menu_cmdline(self, prod=True, file="data/init.csv"):
        while True:
            print("Signal Process menu:")
            print("--------------------")
            print("I: Initialize")
            print("P: Process")
            print("E: Exit")

            init = to_bool(self.config["Signal"]["init"])

            cmd = input("Choice [I/P/E]: ")
            clear()

            if cmd.lower() == "i":
                water_value_pre = input("Watermeter value before (dl): ")
                input("Press Enter when ready...")
                time_start = int(round(time.time() * 1000))
                input("Press Enter when finished...")
                time_end = int(round(time.time() * 1000))
                water_value_post = input("Water value after (dl): ")

                if prod:
                    self.init_config_from_db(time_start, time_end, water_value_pre, water_value_post)
                else:
                    self.init_config_from_file(file, water_value_pre, water_value_post)

                print("config.ini generated successfully")
            elif cmd.lower() == "p" and init:
                if prod:
                    self.process_from_db()
                else:
                    self.process_from_file()
                print("processed signal successfully")
            elif cmd.lower() == "e":
                sys.exit(0)
            else:
                pass

    def init_config_from_db(self, time_start, time_end, water_value_pre, water_value_post):
        pass

    def init_config_from_file(self, file, water_value_pre, water_value_post):
        self.init_class_from_file(file)
        max_p_v = -1
        max_p_i = 0
        i = 0
        for w in self.wave_seg_objects:
            max_tmp_v = len(w.periods)
            if max_p_v < max_tmp_v:
                max_p_v = max_tmp_v
                max_p_i = i

            i += 1

        mean_period = np.mean(self.wave_seg_objects[max_p_i].periods)
        duration = self.wave_seg_objects[max_p_i].duration/1000
        water = (water_value_post - water_value_pre)/10
        debit = water/duration

        self.config["Signal"]["debit"] = str(debit)
        self.config["Signal"]["mean_period"] = str(mean_period)
        self.config["Signal"]["duration"] = str(duration)
        self.config["Signal"]["init"] = "1"

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

    def init_class_from_file(self, file):
        timestamps, data = self.file_get_norms(file)
        data = data - np.mean(data)

        try:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")

            self.data = np.asarray(data)
            self.timestamps = timestamps
            self.noise_det()

            if self.noise_seg_tab:
                self.peak_det()
            if self.min_tab and self.max_tab:
                self.extend_peaks()
            self.compute_periods()

        except (configparser.Error, KeyError):
            print("Invalid config.ini file format")
            sys.exit(1)
        except ValueError:
            print("Invalid parsing from config.ini")
            sys.exit(1)
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

    def process_from_db(self):
        pass

    def process_from_file(self, file):
        self.init_class_from_file(file)
        consom = 0
        x1 = float(self.config["Signal"]["x1"])
        y1 = float(self.config["Signal"]["y1"])
        x2 = float(self.config["Signal"]["x2"])
        y2 = float(self.config["Signal"]["y2"])
        fct, m = self.get_fct(x1, y1, x2, y2)

        for w in self.wave_seg_objects:
            mean_period = np.mean(w.periods)
            duration = w.duration / 1000
            rapport = (mean_period/duration)/ (float(self.config["Signal"]["mean_period"])/float(self.config["Signal"]["duration"]))

            if rapport > 0.01 and float(self.config["Signal"]["sampling_rate"])*2 < mean_period < 350 and duration > 2:
                print("indice_start_tab: " + str(w.istart) + ", indice_end_tab: " + str(w.iend) + ", duration(s): " + str(duration) + ", period: " + str(mean_period))
                # consom += (float(self.config["Signal"]["debit"])/rapport) * duration
                debit = fct(mean_period)
                consom += debit*duration

        print("Tot Consommation (l): " + str(consom))


    # modif min_tab, max_tab
    def peak_det(self):
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        lookformax = True
        first_peak = True  # still looking for first peak

        # looks for type of first peak
        for i in range(0, len(self.data)):
            curr = self.data[i]
            if curr > mx:
                mx = curr
                mxpos = i
            if curr < mn:
                mn = curr
                mnpos = i

            if curr < mx - self.delta:
                lookformax = False
                first_peak = False
                break
            if curr > mn + self.delta:
                lookformax = True
                first_peak = False
                break

        # there is at least 1 peak in this sample
        if not first_peak:
            # end of first noise segment
            i_noise_seg = 1
            # looks for peaks in the signal
            for j in range(i, len(self.data)):
                curr = self.data[j]
                if curr > mx:
                    mx = curr
                    mxpos = j
                    if i_noise_seg < len(self.noise_seg_tab) - 1 and self.noise_seg_tab[i_noise_seg][0] < mxpos:
                        i_noise_seg += 2
                if curr < mn:
                    mn = curr
                    mnpos = j
                    if i_noise_seg < len(self.noise_seg_tab) - 1 and self.noise_seg_tab[i_noise_seg][0] < mnpos:
                        i_noise_seg += 2

                if lookformax:
                    if curr < mx - self.delta:
                        pos = mxpos
                        if pos in range(self.noise_seg_tab[i_noise_seg-1][0], self.noise_seg_tab[i_noise_seg][0]+1):
                            pos = self.noise_seg_tab[i_noise_seg-1][0]
                        self.max_tab.append((pos, self.data[pos]))
                        mn = curr
                        mnpos = j
                        lookformax = False

                else:
                    if curr > mn + self.delta:
                        pos = mnpos
                        if pos in range(self.noise_seg_tab[i_noise_seg-1][0], self.noise_seg_tab[i_noise_seg][0]):
                            pos = self.noise_seg_tab[i_noise_seg-1][0]
                        self.min_tab.append((pos, self.data[pos]))
                        mx = curr
                        mxpos = j
                        lookformax = True

    @staticmethod
    def peakdet_basic(v, delta, x=None):
        maxtab = []
        mintab = []

        if x is None:
            x = np.arange(len(v))

        v = np.asarray(v)

        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')

        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')

        if delta <= 0:
            sys.exit('Input argument delta must be positive')

        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN

        lookformax = True

        for i in np.arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]

            if lookformax:
                if this < mx - delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        return np.array(maxtab), np.array(mintab)

    # modif noise_seg_tab, noise_tab
    def noise_det(self):
        last_noise_pos = -math.inf
        start_of_seg = True

        # check slope of the "width" next values
        for i in range(0, len(self.data)-self.seg_width):
            # get slope
            x1 = i
            y1 = np.mean(self.data[i:i + self.seg_width // 2])
            x2 = i + self.seg_width
            y2 = np.mean(self.data[i + self.seg_width // 2:i + self.seg_width * 2])
            f, m = self.get_fct(x1, y1, x2, y2)

            # add end of segment
            # does not start segment and (does not directly follow a noise or is last point)
            if not start_of_seg and (i > last_noise_pos or i == len(self.data)-self.seg_width-1):
                self.noise_seg_tab.append((last_noise_pos, self.data[last_noise_pos]))
                start_of_seg = True

            # add noise segment if slope low enough
            if -self.slope < m < self.slope \
                    and max(self.data[i:i + self.seg_width]) - min(self.data[i:i + self.seg_width]) < 2:
                # check last noise segment to not add redundant information
                noise_start = max(last_noise_pos, i)
                for j in range(noise_start, i + self.seg_width + 1):
                    self.noise_tab.append((j, self.data[j]))
                last_noise_pos = i + self.seg_width

                if start_of_seg:
                    self.noise_seg_tab.append((i, self.data[i]))
                    start_of_seg = False

        i_seg_s = 2
        i_seg_e = 1
        while i_seg_s < len(self.noise_seg_tab):
            seg_a_end = self.noise_seg_tab[i_seg_e][0]
            seg_b_start = self.noise_seg_tab[i_seg_s][0]
            between_a_b = self.data[seg_a_end:seg_b_start]

            if max(between_a_b) - min(between_a_b) < float(self.config["Signal"]["max_peak"])/4:
                self.noise_seg_tab.pop(i_seg_e)
                self.noise_seg_tab.pop(i_seg_e)
            else:
                i_seg_e += 2
                i_seg_s += 2

        i_seg = 0
        while i_seg < len(self.noise_seg_tab):
            if self.noise_seg_tab[i_seg+1][0] - self.noise_seg_tab[i_seg][0] <= 100:
                self.noise_seg_tab.pop(i_seg)
                self.noise_seg_tab.pop(i_seg)
            else:
                i_seg += 2

    def extend_peaks(self):
        # there is noise before the firsts peaks
        if self.noise_seg_tab[1] < self.max_tab[0] and self.noise_seg_tab[1] < self.min_tab[0]:
            self.wave_seg_tab.extend(self.noise_seg_tab[1:-1])
        else:
            self.wave_seg_tab.append((0, 0))
            self.wave_seg_tab.extend(self.noise_seg_tab[:-1])

        # there is noise after last peaks
        if self.noise_seg_tab[-1] > self.max_tab[-1] and self.noise_seg_tab[-1] > self.min_tab[-1]:
            pass
        else:
            self.wave_seg_tab.append(self.noise_seg_tab[-1])
            self.wave_seg_tab.append((len(self.data)-1, self.data[-1]))

    # period in ms or samples?
    def compute_periods(self):
        crossed_max_min = self.cross_concatenation(self.max_tab, self.min_tab)
        i_seg = 0
        i_mx_mn = 0
        while i_seg < len(self.wave_seg_tab):
            cnt = 0
            while i_mx_mn < len(crossed_max_min):
                if crossed_max_min[i_mx_mn] <= self.wave_seg_tab[i_seg+1]:
                    i_mx_mn += 1
                    cnt += 1
                else:
                    break
            start = self.wave_seg_tab[i_seg]
            end = self.wave_seg_tab[i_seg + 1]
            min_max = crossed_max_min[i_mx_mn-cnt:i_mx_mn]
            self.wave_seg_objects.append(self.WaveSegment(start, end, min_max, self.timestamps, float(self.config["Signal"]["max_peak"])))

            i_seg += 2

        # for seg in self.wave_seg_objects:
        #     print(seg.min_max)

    class WaveSegment:
        def __init__(self, start, end, min_max, timestamps, peakheight):
            self.start = timestamps[start[0]]
            self.istart = start[0]
            self.iend = end[0]
            self.duration = timestamps[end[0]] - self.start
            self.periods = []

            # s_min_max = 0
            # while s_min_max < len(min_max) and self.start[0] > s_min_max:
            #     s_min_max += 1
            #
            # e_min_max = s_min_max
            #
            # while e_min_max < len(min_max) and self.end[0] > e_min_max:
            #     e_min_max += 1
            #
            # min_max = min_max[s_min_max:e_min_max]

            if len(min_max) == 0:
                period = 0
                self.periods.append(period)
            elif len(min_max) == 1:
                start_to_peak = np.abs(min_max[0][0] - start[0])
                peak_to_end = np.abs(end[0] - min_max[0][0])
                if start_to_peak < peakheight/4:
                    if peak_to_end < peakheight/4:
                        period = 0
                    else:
                        period = (peak_to_end*2)*2  # rough estimation, no great impact on consumption
                else:
                    period = (start_to_peak*2)*2

                self.periods.append(period)
            else:
                for i in range(0, len(min_max)-1):
                    period = (min_max[i+1][0] - min_max[i][0])*2
                    self.periods.append(period)

    @staticmethod
    def get_fct( x1, y1, x2, y2):
        try:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
        except ZeroDivisionError:
            print("division by zero for get_fct")

            def linear_fct(x):
                return -1

            return linear_fct

        def linear_fct(x):
            return m * x + b

        return linear_fct, m

    @staticmethod
    def sine_fun( x, a, b):
        return a*np.sin(b*x)

    @staticmethod
    def cross_concatenation(la, lb):
        res = []

        if len(la) >= len(lb):
            pass # OK
        else:
            la, lb = lb, la

        for i in range(0, len(lb)):
            res.append(la[i])
            res.append(lb[i])

        if len(la) > len(lb):
            res.append(la[-1])

        return res

    @staticmethod
    def file_get_norms(file, it=0):
        with open(file, "r") as input_file:
            data_lines = input_file.readlines()

            timestamps = []
            values = []

            for line in data_lines:
                l = line.split(",")
                timestamps.append(float(l[0]))
                values.append(float(l[1][:-1]))

            return timestamps, values

    @staticmethod
    def db_get_norms(self, _n):
        try:
            try:
                db_config = self.config["Database"]
                conn = psycopg2.connect(
                    user=db_config["user"],
                    password=db_config["user"],
                    host=db_config["user"],
                    port=db_config["user"],
                    database=db_config["user"]
                )
            except configparser.Error:
                message = "Error while loading database config file"
                raise
            except Exception:
                message = "Error db conn"
                raise

            try:
                curs = conn.cursor()
                curs.execute("SELECT value FROM water_consumption ORDER BY timestamp DESC LIMIT %(n)s;", {"n": _n})

                a = []
                row = curs.fetchone()
                while row is not None:
                    a.append(row[0])

                    row = curs.fetchone()

                curs.close()
                return a
            except Exception:
                message = "Error cursor db"
                raise
        except Exception as e:
            print(message + str(e))


def to_bool(v):
    return str(v).lower() in ("yes", "y", "oui", "true", "t", "1")


def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
        # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')


def main():
    signal = SignalProcessing()
    # signal.menu_cmdline(False)
    # signal.init_config_from_file("data/init.csv", 13684406, 13684429)
    signal.process_from_file("data/excel.csv")

    plt.plot(signal.data)

    # print("noisetab: " + str(signal.noise_tab))
    # print("noise seg: " + str(signal.noise_seg_tab))
    # if signal.noise_seg_tab:
    #     plt.scatter(np.array(signal.noise_seg_tab)[:, 0], np.array(signal.noise_seg_tab)[:, 1], color='black')

    # if signal.max_tab:
    #     plt.scatter(np.array(signal.max_tab)[:, 0], np.array(signal.max_tab)[:, 1], color='blue')
    # if signal.min_tab:
    #     plt.scatter(np.array(signal.min_tab)[:, 0], np.array(signal.min_tab)[:, 1], color='red')

    if signal.wave_seg_tab:
        plt.scatter(np.array(signal.wave_seg_tab)[:, 0], np.array(signal.wave_seg_tab)[:, 1], color='black')

    plt.show()


if __name__ == "__main__":
    main()

