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

    def _linear_fit(self, x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, intercept, r_value, p_value, std_err

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

            zero_order_test = self._linear_fit(
                time, concentration)
            first_order_test = self._linear_fit(
                time, np.log(concentration))
            second_order_test = self._linear_fit(time, 1/concentration)

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

    def _plot_params(self, ax=None, plot_type='conc', time_unit='second',
                     formula='A', conc_unit='mol/L', size=12,
                     conc_or_p='conc'):
        linewidth = 2

        # grid and ticks settings
        ax.minorticks_on()
        ax.grid(b=True, which='major', linestyle='--',
                linewidth=linewidth - 0.5)
        ax.grid(b=True, which='minor', axis='x',
                linestyle=':', linewidth=linewidth - 1)
        ax.tick_params(which='both', labelsize=size+2)
        ax.tick_params(which='major', length=6, axis='both')
        ax.tick_params(which='minor', length=3, axis='both')

        ax.set_xlabel('Time / {}'.format(time_unit), size=size+3)

        label_formula = re.sub("([0-9])", "_\\1", formula)

        if conc_or_p == 'conc':
            label_formula = '$\mathregular{['+label_formula+']}$'
        elif conc_or_p == 'p':
            label_formula = r'$\mathregular{P_{'+label_formula+r'}}$'

        if plot_type == 'conc':
            ax.set_ylabel('{0} / {1}'.format(label_formula, conc_unit),
                          size=size+3)
        elif plot_type == 'ln_conc':
            ax.set_ylabel('ln({0})'.format(label_formula),
                          size=size+3)
        elif plot_type == 'inv_conc':
            ax.set_ylabel('1/{0}'.format(label_formula) + r' / $\mathregular{{({0})^{{-1}}}}$'.format(conc_unit),  # NoQA
                          size=size+3)
        else:
            raise ValueError('Plot type not valid')

        return

    def _xy(self, plot_type='conc', column='average'):

        x = self.data_unc[column]
        y = self.data_unc['concentration']

        if plot_type == 'conc':
            y = y
        elif plot_type == 'ln_conc':
            y = unumpy.log(y)
        elif plot_type == 'inv_conc':
            y = 1/(y)
        else:
            raise ValueError('Plot type not valid')

        x_values = unumpy.nominal_values(x)
        y_values = unumpy.nominal_values(y)

        x_err = unumpy.std_devs(x)
        y_err = unumpy.std_devs(y)

        return x_values, x_err, y_values, y_err

    def plot(self, size=(8, 6), plot_type='conc', ax=None, time_unit='second',
             formula='A', conc_unit='mol/L', conc_or_p='conc',
             linear_fit=False, column='average'):

        if ax is None:
            fig, ax = plt.subplots(figsize=size, facecolor=(1.0, 1.0, 1.0))

        self._plot_params(ax, plot_type=plot_type, time_unit=time_unit,
                          formula=formula, conc_unit=conc_unit,
                          conc_or_p=conc_or_p)

        x_values, x_err, y_values, y_err = self._xy(plot_type, column)

        ax.errorbar(x_values, y_values, fmt='ro', xerr=x_err,
                    yerr=y_err, ecolor='k', capsize=3)

        if linear_fit:
            slope, intercept, r_value, p_value, std_err = self._linear_fit(
                x_values, y_values)
            ax.plot(x_values, slope * x_values + intercept,
                    label='y={:.2E}x{:+.2E}  $R^2= {:.2f}$'.format(slope,
                                                                   intercept,
                                                                   r_value**2))

            ax.legend(loc='best', fontsize=14)

        return ax