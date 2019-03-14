from pathlib import Path

import pandas as pd

__all__ = ['load_etf']

data_dir: Path = Path(__file__).parent.joinpath('data')


def load_etf(typ='returns') -> pd.DataFrame:
    """
    Monthly returns or price from 2010-09-01 to 2019-03-01 of

    * Vanguard S&P 500 ETF (VOO)
    * iShares MSCI Emerging Markets ETF
    * Vanguard Total World Stock ETF

    By default, function gives returns data, which has one last day of data (first day is NA).

    Parameters
    ----------
    typ: {'returns', 'price'}
        Determines whether price or returns data is required

    Returns
    -------
    DataFrame
        DataFrame containing monthly price or returns data with the date as the index
    """
    fp = data_dir.joinpath('etf.csv')
    df = pd.read_csv(fp, parse_dates=['Date'], index_col=0)

    if typ == 'price':
        return df
    else:
        return df.pct_change().dropna()
