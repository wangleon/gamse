import re
import numpy as np
from astropy.table import Table

class FilterableTable(Table):

    def __init__(self, *args, **kwargs):
        """
        Initialize FilterableTable.
        """

        super().__init__(*args, **kwargs)

    def filter(self, conditions, case_sensitive=False):
        """
        Filter the obslogtable using condtions.

        Args:
            condtions: dict

        Returns:
            FilterableTable

        """
        if not conditions:
            return self.copy()

        mask = np.ones(len(self), dtype=bool)

        for column, condition in conditions.items():
            if column not in self.colnames:
                # skip if column does not exist
                continue

            col_data = self[column]
            col_mask = self._get_column_mask(col_data, condition,
                                             case_sensitive=case_sensitive)

            if col_mask is not None:
                mask &= col_mask

        return self[mask]

    def _get_column_mask(self, col_data, condition, case_sensitive):
        """
        Args:

        """
        if isinstance(condition, (list, tuple, np.ndarray)):
            return self._handle_in_condition(col_data, condition, exclude=False)

        if isinstance(condition, str):
            return self._parse_string_condition(col_data, condition,
                                                case_sensitive)

        if callable(condition):
            return self._handle_callable_condition(col_data, condition)

        # default is "equalization"
        return col_data == condition

    def _parse_string_condition(self, col_data, condition_str, case_sensitive):
        """

        """

        # remove blank strings
        condition_str = condition_str.strip()

        # check if it's a regular expression
        if condition_str.startswith('~'):
            pattern = condition_str[1:]
            try:
                regex = re.compile(pattern)
                if coldata.dtype.kind in ('U', 'S'):
                    mask = [bool(regex.search(str(val))) for val in col_data]
                    return np.array(mask)
            except re.error:
                pass

        # check:
        if ':' in condition_str and not condition_str.startswith((
            'has:', 'in:', 'not in:')):
            parts = condition_str.split(':')
            if len(parts) == 2:
                start, end = parts
                mask = np.ones(len(col_data), dtype=bool)
                if start:
                    mask &= col_data >= self._parse_value(start, col_data.dtype)
                if end:
                    mask &= col_data <= self._parse_value(end, col_data.dtype)
                return mask

        operators = [
            ('>=', self._handle_greater_equal),
            ('<=', self._handle_less_equal),
            ('!=', self._handle_not_equal),
            ('>',  self._handle_greater_than),
            ('<',  self._handle_less_than),
            ('=',  self._handle_equal),
            ('has:', self._handle_contains),
            ('in:', self._handle_in_list),
            ('not in:', self._handle_not_in_list),
        ]
        
        for operator, handler in operators:
            if condition_str.startswith(operator):
                value_str = condition_str[len(operator):]
                value = self._parse_value(value_str, col_data.dtype)
                return handler(col_data, value, case_sensitive)

        # default: string equality
        if col_data.dtype.kind in ('U', 'S'):
            if case_sensitive:
                return col_data == condition_str
            else:
                return np.char.lower(col_data.astype(str)) == condition_str.lower()
        else:
            # try to convert to float
            try:
                value = float(condition_str) if '.' in condition_str else int(condition_str)
                return col_data == value
            except (ValueError, TypeError):
                return col_data == condition_str

    def _parse_value(self, value_str, dtype):
        """
        """
        if dtype is None:
            try:
                return float(value_str) if '.' in value_str else int(value_str)
            except ValueError:
                return value_str

        kind = dtype.kind

        if kind in ('i', 'u'):
            return int(value_str)
        elif kind in ('f', 'c'):
            return float(value_str)
        elif kind == 'b':
            return value_str.lower() in ('true', '1', 'yes', 't')
        elif kind in ('U', 'S'):
            return value_str
        else:
            return value_str

    def _handle_equal(self, col_data, value, case_sensitive=False):
        if col_data.dtype.kind in ('U', 'S'):
            # for string type
            # use case-insensitive comparison
            if isinstance(value, str):
                if case_sensitive:
                    return col_data == value
                else:
                    return np.char.lower(col_data.astype(str)) == str(value).lower()
            else:
                # if value is not a string, compare as-is
                return col_data == value
        else:
            # for non-string types, use standard comparison
            return col_data == value

    def _handle_not_equal(self, col_data, value, case_sensitive=False):
        """Not equal operation
        """
        if col_data.dtype.kind in ('U', 'S'):
            if isinstance(value, str):
                if case_sensitive:
                    return col_data != value
                else:
                    return np.char.lower(col_data.astype(str)) != str(value).lower()
            else:
                # if value is not a string, compare as-is
                return col_data != value
        else:
            # for non-string types, use standard comparison
            return col_data != value

    def _handle_greater_than(self, col_data, value, case_sensitive=False):
        try:
            return col_data > value
        except TypeError:
            mask = [str(x) > str(value) for x in col_data]
            return np.array(mask)

    def _handle_less_than(self, col_data, value, case_sensitive=False):
        try:
            return col_data < value
        except TypeError:
            mask = [str(x) < str(value) for x in col_data]
            return np.array(mask)

    def _handle_greater_equal(self, col_data, value, case_sensitive=False):
        try:
            return col_data >= value
        except TypeError:
            mask = [str(x) >= str(value) for x in col_data]
            return np.array(mask)

    def _handle_less_equal(self, col_data, value, case_sensitive=False):
        try:
            return col_data <= value
        except TypeError:
            mask = [str(x) <= str(value) for x in col_data]
            return np.array(mask)

    def _handle_contains(self, col_data, value, case_sensitive=False):
        if col_data.dtype.kind in ('U', 'S'):
            # for string type
            if case_sensitive:
                parsed_col_data = col_data
                parsed_value = str(value)
            else:
                parsed_col_data = np.char.lower(col_data.astype(str))
                parsed_value = str(value).lower()

            idx = np.char.find(parsed_col_data, parsed_value)
            return idx >= 0

        else:
            # for non-string datatypes, convert to strings
            if case_sensitive:
                mask = [str(val).find(str(value)) >=0
                        for val in col_data]
            else:
                mask = [str(val).lower().find(str(value).lower()) >=0
                        for val in col_data]
            return np.array(mask)

    def _handle_in_list(self, col_data, value, case_sensitive=False):
        if isinstance(value, str):
            items = [item.strip() for item in value.split(',')]
            # try to convert to numeric values
            proc_items = []
            for item in items:
                try:
                    proc_items.append(float(item) if '.' in item else int(item))
                except ValueError:
                    proc_items.append(item)
            value = proc_items

        if col_data.dtype.kind in ('U', 'S'):
            # for string type
            if case_sensitive:
                return np.isin(col_data, value)
            else:
                # case-insensitive in list
                # convert all list items to lowercase if they are strings
                lower_value = [item.lower()
                               if isinstance(item, str) else item
                               for item in value]
                # convert column data to lower case
                lower_col_data = np.char.lower(col_data.astype(str))

                return np.isin(lower_col_data, lower_value)
        else:
            # for non-string types
            return np.isin(col_data, value)

    def _handle_not_in_list(self, col_data, value, case_sensitive=False):
        if isinstance(value, str):
            items = [item.strip() for item in value.split(',')]
            # try to conver to numeric values
            proc_items = []
            for item in items:
                try:
                    proc_items.append(float(item) if '.' in item else int(item))
                except ValueError:
                    proc_items.append(item)
            value = proc_items

        if col_data.dtype.kind in ('U', 'S'):
            # for string type
            if case_sensitive:
                return ~np.isin(col_data, value)
            else:
                # case-insensitive in list
                # convert all list items to lowercase if they are strings
                lower_value = [item.lower()
                               if isinstance(item, str) else item
                               for item in value]
                # convert column data to lower case
                lower_col_data = np.char.lower(col_data.astype(str))

                return ~np.isin(lower_col_data, lower_value)
        else:
            # for non-string types
            return ~np.isin(col_data, value)

    def _handle_in_condition(self, col_data, value_list, exclude=False):
        if exclude:
            return ~np.isin(col_data, value_list)
        else:
            return np.isin(col_data, value_list)

    def _handle_callable_condition(self, col_data, func):
        return np.array([func(val) for val in col_data])

    @classmethod
    def read(cls, *args, **kwargs):
        """
        Override the read() method to use 'ascii.fixed_width_two_line' as the
        default format.

        Args:
            *args:
            **kwargs: 

        Returns:
            FilterableTable:

        """
        if 'format' not in kwargs:
            kwargs['format'] = 'ascii.fixed_width_two_line'

        table = super().read(*args, **kwargs)
        return cls(table)

    def write(self, *args, **kwargs):
        """
        Override the write() method to use 'ascii.fixed_width_two_line' as the
        default format.

        Args:

        Returns:
            None

        """
        if 'format' not in kwargs:
            kwargs['format'] = 'ascii.fixed_width_two_line'

        return super().write(*args, **kwargs)

