import numpy as np

class SLIMCoefficientConstraints:
    def __init__(self, **kwargs):
        if 'variable_names' in kwargs:
            variable_names = kwargs.get('variable_names')
            P = len(variable_names)
        elif 'P' in kwargs:
            P = kwargs.get('P')
            variable_names = ["x_" + str(i) for i in range(1, P + 1)]
        else:
            raise ValueError("user needs to provide 'P' or 'variable_names'")

        self.P = P
        self.variable_names = variable_names
        self.fix_flag = kwargs.get('fix_flag', True)
        self.check_flag = kwargs.get('check_flag', True)
        self.print_flag = kwargs.get('print_flag', False)

        ub = kwargs.get('ub', 10.0 * np.ones(P))
        lb = kwargs.get('lb', -10.0 * np.ones(P))
        vtype = kwargs.get('type', ['I'] * P)
        C_0j = kwargs.get('C0_j', np.nan * np.ones(P))
        sign = kwargs.get('sign', np.nan * np.ones(P))

        self.ub = self._check_numeric_input('ub', ub)
        self.lb = self._check_numeric_input('lb', lb)
        self.C_0j = self._check_numeric_input('C_0j', C_0j)
        self.sign = self._check_numeric_input('sign', sign)
        self.vtype = self._check_string_input('vtype', vtype)

        if self.check_flag:
            self.check_set()
        if self.print_flag:
            self.view()

    def _check_string_input(self, input_name, input_value):
        if isinstance(input_value, np.ndarray):
            if input_value.size == self.P:
                return input_value
            elif input_value.size == 1:
                return np.repeat(input_value, self.P)
            else:
                raise ValueError(f"length of {input_name} is {input_value.size}; should be {self.P}")
        elif isinstance(input_value, str):
            return np.array([input_value] * self.P, dtype=str)
        elif isinstance(input_value, list):
            if len(input_value) == self.P:
                return np.array([str(x) for x in input_value])
            elif len(input_value) == 1:
                return np.repeat(input_value, self.P)
            else:
                raise ValueError(f"length of {input_name} is {len(input_value)}; should be {self.P}")
        else:
            raise ValueError(f"unsupported type for {input_name}")

    def _check_numeric_input(self, input_name, input_value):
        if isinstance(input_value, np.ndarray):
            if input_value.size == self.P:
                return input_value
            elif input_value.size == 1:
                return input_value * np.ones(self.P)
            else:
                raise ValueError(f"length of {input_name} is {input_value.size}; should be {self.P}")
        elif isinstance(input_value, (float, int)):
            return float(input_value) * np.ones(self.P)
        elif isinstance(input_value, list):
            if len(input_value) == self.P:
                return np.array([float(x) for x in input_value])
            elif len(input_value) == 1:
                return np.array([float(input_value[0])] * self.P)
            else:
                raise ValueError(f"length of {input_name} is {len(input_value)}; should be {self.P}")
        else:
            raise ValueError(f"unsupported type for {input_name}")

    def check_set(self):
        for i in range(len(self.variable_names)):
            if self.ub[i] < self.lb[i]:
                self.ub[i], self.lb[i] = self.lb[i], self.ub[i]

            if self.sign[i] > 0 and self.lb[i] < 0:
                self.lb[i] = 0.0

            if self.sign[i] < 0 and self.ub[i] > 0:
                self.ub[i] = 0.0

            if self.variable_names[i] in {'Intercept', '(Intercept)', 'intercept', '(intercept)'}:
                if self.C_0j[i] > 0 or np.isnan(self.C_0j[i]):
                    self.C_0j[i] = 0.0

    def view(self):
        x = PrettyTable()
        x.align = "r"
        x.add_column("variable_name", self.variable_names)
        x.add_column("vtype", list(self.vtype))
        x.add_column("sign", list(self.sign))
        x.add_column("lb", list(self.lb))
        x.add_column("ub", list(self.ub))
        x.add_column("C_0j", list(self.C_0j))
        print(x)
