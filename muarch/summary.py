from typing import Collection

from statsmodels.iolib.summary import Summary as S


class Summary(S):
    def __init__(self, smry: S):
        self.header = None
        self.tables = smry.tables
        self.extra_txt = smry.extra_txt

    def add_header(self, header):
        self.header = header

    def as_text(self):
        """return tables as string

        Returns
        -------
        txt : string
            summary tables and extra text as one string

        """
        txt = super().as_text()
        if self.header:
            txt = f'{self.header}\n\n{txt}'
        return txt

    def as_latex(self):
        """return tables as string

        Returns
        -------
        latex : string
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.

        """
        latex = super().as_latex()
        if self.header:
            latex = f'{self.header}\n\n{latex}'
        return latex

    def as_csv(self):
        """return tables as string

        Returns
        -------
        csv : string
            concatenated summary tables in comma delimited format

        """
        csv = super().as_csv()
        if self.header:
            csv = f'{self.header}\n\n{csv}'
        return csv

    def as_html(self):
        """return tables as string

        Returns
        -------
        html : string
            concatenated summary tables in HTML format

        """
        html = super().as_html()
        if self.header:
            html = f'<h2>{self.header}</h2>{html}'
        return html


class SummaryList:
    def __init__(self, summaries: Collection[Summary] = ()):
        self._summaries = list(summaries)

    @staticmethod
    def _assert_is_summary(obj):
        if not isinstance(obj, Summary):
            raise ValueError('object does not subclass Summary')

    def __getitem__(self, i):
        return self._summaries[i]

    def __setitem__(self, i, smry: Summary):
        self._assert_is_summary(smry)
        self._summaries[i] = smry

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        # return '<' + str(type(self)) + '>\n"""\n' + self.__str__() + '\n"""'
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        """Display as HTML in IPython notebook."""
        return self.as_html()

    def append(self, smry: Summary):
        self._assert_is_summary(smry)
        self._summaries.append(smry)

    def extend(self, summaries: Collection[Summary]):
        for s in summaries:
            self._assert_is_summary(s)
        self._summaries.extend(summaries)

    def as_text(self):
        """return tables as string

        Returns
        -------
        txt : string
            summary tables and extra text as one string

        """
        sep = '\n\n' + '*' * 100 + '\n\n'
        return sep.join(s.as_text() for s in self)

    def as_latex(self):
        """return tables as string

        Returns
        -------
        latex : string
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.

        """
        sep = '\n\n' + '*' * 100 + '\n\n'
        return sep.join(s.as_latex() for s in self)

    def as_csv(self):
        """return tables as string

        Returns
        -------
        csv : string
            concatenated summary tables in comma delimited format

        """
        sep = '\n\n' + '*' * 100 + '\n\n'
        return sep.join(s.as_csv() for s in self)

    def as_html(self):
        """return tables as string

        Returns
        -------
        html : string
            concatenated summary tables in HTML format

        """

        sep = '<br/>' * 2 + '<hr/>' + '<br/>' * 2
        return sep.join(s.as_html() for s in self)
