import numpy as np
import pandas as pd
from uncertainties import unumpy
import matplotlib.pyplot as plt
from scipy import stats
import re


class KineticsAnalyser:
    def __init__(self, data_file, skip_rows=0):
        self.data_file = pd.read_csv(data_file, skiprows=skip_rows)

        self.dilutions = self.data_file.shape[0]

        self.experiments = divmod(self.data_file.shape[1]-1, 2)[0]

        def _data_unc(self):
            df = pd.DataFrame()
            df['concentration'] = unumpy.uarray(
                self.data_file.iloc[:, 0], self.data_file.iloc[:, 1])

            exp_number = 1
            column = 2

            for i in range(self.experiments):
                column_name = 'Experiment ' + str(exp_number)
                df[column_name] = unumpy.uarray(
                    self.data_file.iloc[:, column], self.data_file.iloc[:, column+1])
                exp_number += 1
                column += 2

            # df['average'] = df.iloc[:, 1:].mean(axis=1)
            # df['median'] = df.iloc[:, 1:].median(axis=1)
            # the above code lines do not work due to a bug: https://github.com/pandas-dev/pandas/issues/14162

            df['average'] = df.iloc[:, 1:].apply(
                lambda x: x.sum() / x.size, axis=1)
            df['median'] = df.iloc[:, 1:-1].apply(
                lambda x: np.median(x), axis=1)

            return df

        self.data_unc = _data_unc(self)

    def time_dispersion(self):
        return self.data_unc.iloc[:, 1:-2].apply(lambda x: np.std(unumpy.nominal_values(x), ddof=1), axis=1)

    def summary(self):
        concentration = unumpy.nominal_values(
            self.data_unc['concentration'])

        d = {}

        for column in self.data_unc.iloc[:, 1:]:
            v = []
            d_order = {}

            time = unumpy.nominal_values(
                self.data_unc[column])

            zero_order_test = stats.linregress(
                time, concentration)
            first_order_test = stats.linregress(
                time, np.log(concentration))
            second_order_test = stats.linregress(time, 1/concentration)

            d_order['R2_zero_order'] = zero_order_test[2]**2
            d_order['R2_first_order'] = first_order_test[2]**2
            d_order['R2_second_order'] = second_order_test[2]**2

            if max(d_order, key=d_order.get) == 'R2_zero_order':
                order = 0
                rate_constant = unumpy.uarray(
                    abs(zero_order_test[0]), zero_order_test[-1])
            elif max(d_order, key=d_order.get) == 'R2_first_order':
                order = 1
                rate_constant = unumpy.uarray(
                    abs(first_order_test[0]), first_order_test[-1])
            else:
                order = 2
                rate_constant = unumpy.uarray(
                    abs(second_order_test[0]), second_order_test[-1])

            v = [d_order['R2_zero_order'], d_order['R2_first_order'],
                 d_order['R2_second_order'], order, rate_constant]

            d[column] = v

        df = pd.DataFrame(d, index=['R2 zero order', 'R2 first order',
                                    'R2 second order', 'Order',
                                    'Rate constant'])
        return df
