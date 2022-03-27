import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

from decimal import Decimal, localcontext, ROUND_DOWN

def truncate(number, places):
    if not isinstance(places, int):
        raise ValueError("Decimal places must be an integer.")
    if places < 1:
        raise ValueError("Decimal places must be at least 1.")
    # If you want to truncate to 0 decimal places, just do int(number).

    with localcontext() as context:
        context.rounding = ROUND_DOWN
        exponent = Decimal(str(10 ** - places))
        return Decimal(str(number)).quantize(exponent).to_eng_string()

value = 1.232847632487
trunc_value = float(truncate(value, 4))
print(value, trunc_value)
