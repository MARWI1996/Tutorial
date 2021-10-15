import numpy as np
import sqlalchemy as db
import os
import matplotlib.pyplot as plt


class Data(object):
    '''
    Data object which is handling the provided data in .csv-format
    '''

    def __init__(self, filename):
        '''
        class constructor
        '''
        self.data = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
        self.x_data = self.data[:, 0]
        self.num_rows, self.num_cols = self.data.shape

    def return_x_data(self):
        '''
        returns the x data of the Data object
        '''
        return self.x_data

    def return_y_data(self, i=1):
        '''
        returns the y data of the Data object by defining the index "i" as the row
        '''
        self.i = i

        try:
            return self.data[:, int(self.i)]
        except:
            print(f"wrong input! Please only integers from 1 to {self.num_cols - 1} allowed")

    def return_num_rows(self):
        '''
        returns the number of rows of the Data object
        '''
        return self.num_rows

    def return_num_cols(self):
        '''
        returns the number of columns of the Data object
        '''
        return self.num_cols

    def return_xy_data(self, row, column):
        '''
        returns one specific value of the dataset, by defining the row and column
        '''
        self.row = row
        self.column = column
        return self.data[self.row, self.column]


class DataNotIdeal(Data):
    '''
    Inhertated Data object for handling the not ideal data, with more specialized methods
    '''

    def __init__(self, filename):
        '''
        class constructor of the heritance
        '''
        super().__init__(filename)
        self.assigned_ideal_func = []
        self.ls_error = []
        self.max_error = []

    def assign_ideal_func(self, index_ideal_func, ls_error, max_error):
        '''
        assign the found ideal function (index, least square error and max error) to the data set
        '''
        self.assigned_ideal_func.append(index_ideal_func)
        self.ls_error.append(ls_error)
        self.max_error.append(max_error)

    def return_ideal_func_index(self):
        '''
        return the assigned ideal function
        '''
        return self.assigned_ideal_func

    def return_ideal_func_error_ls(self):
        '''
        return the determined least square error sum
        '''
        return self.ls_error

    def return_ideal_func_max_error(self):
        '''
        return the determined max error for the mapping citerion
        '''
        return self.max_error


# test data
test_data = DataNotIdeal("test.csv")
# train data
train_data = DataNotIdeal("train.csv")
# ideal data
ideal_data = Data("ideal.csv")


def find_ideal_func():
    '''
    finding the ideal functions with the use of the train functions
    calculation of the max. acceptable error for the mapping criterion and assigning to the train data
    '''
    # initialization of variables
    TOTAL_ERROR_temp = 0.0
    TOTAL_ERROR = None
    MAX_error_temp = 0
    MAX_error = 0
    ideal_index = None

    # each column of train data
    for num in range(1, train_data.return_num_cols()):
        # each column of ideal data
        for i_columns in range(1, ideal_data.return_num_cols()):
            # compare each entry between ideal and train function
            for i_rows in range(0, ideal_data.return_num_rows()):
                TOTAL_ERROR_temp = TOTAL_ERROR_temp + (
                            ideal_data.return_xy_data(i_rows, i_columns) - train_data.return_xy_data(i_rows, num)) ** 2
            if (TOTAL_ERROR == None) or (TOTAL_ERROR_temp <= TOTAL_ERROR):
                TOTAL_ERROR = TOTAL_ERROR_temp
                ideal_index = i_columns
            else:
                pass

            # reseting the temporary total error after comparing the train function with one ideal function
            TOTAL_ERROR_temp = 0

        # determine max error between each train function and determined ideal function
        for m in range(0, ideal_data.num_rows):
            MAX_error_temp = ideal_data.return_xy_data(m, ideal_index) - train_data.return_xy_data(m, num)

            if (MAX_error == None) or (MAX_error_temp > MAX_error):
                MAX_error = MAX_error_temp
            else:
                pass

        # assign the determined index, lsq error and max. deviation from ideal function to the train object
        train_data.assign_ideal_func(ideal_index, TOTAL_ERROR, MAX_error)

        # reset of variables after each train data column
        TOTAL_ERROR = None
        MAX_error_temp = 0
        MAX_error = 0
        ideal_index = None

# execute the function
find_ideal_func()

# export the determined indices of the ideal functions
ideal_funcs_indices = train_data.return_ideal_func_index()
ideal_funcs_errors = train_data.return_ideal_func_max_error()


def test_mapping():
    '''
    mapping each data point of the test function to one of the 4 determined ideal functions
    If there are more than one possible mappings, the one with the smaller error will be selected
    '''
    # variable definition
    mapped_function = []
    mapped_smallest_error = []
    i_temp = None
    error_temp = 0
    error_lowest = 0

    # each row of the test data
    for i_row_test in range(0, test_data.return_num_rows()):
        # finding the correct x-value from the determined 4 ideal functions
        for i_row_ideal in range(0, ideal_data.return_num_rows()):
            if (test_data.return_xy_data(i_row_test, 0) == ideal_data.return_xy_data(i_row_ideal, 0)):
                # check for each determined ideal function if the mapping criterion is valid and if more than 1 mapping is possible
                for i in range(0, len(ideal_funcs_indices)):
                    error_temp = abs(
                        ideal_data.return_xy_data(i_row_ideal, ideal_funcs_indices[i]) - test_data.return_xy_data(
                            i_row_test, 1))
                    # first ideal function
                    if i == 0:
                        error_lowest = error_temp
                        if error_lowest <= ideal_funcs_errors[i]:
                            i_temp = i
                    # rest of the ideal functions
                    elif (i != 0) and (error_temp < error_lowest):
                        error_lowest = error_temp
                        if error_lowest <= ideal_funcs_errors[i]:
                            i_temp = i

                mapped_function.append(i_temp)
                mapped_smallest_error.append(error_lowest)

                # reset of variables
                i_temp = None
                error_temp = 0
                error_lowest = 0

    return (mapped_function, mapped_smallest_error)

(mapping, mapped_smallest_error)  = test_mapping()


def create_database():
    '''
    Function for creating the database where the data of the ideal functions, test functions
    and the found mapping between test function and determined ideal functions will be stored.
    With the same function the database will be loaded with the relevant data.
    '''
    # general settings
    BASE_DIR = os.getcwd()
    connection_string = "sqlite:///" + os.path.join(BASE_DIR, 'DATA.db')
    engine = db.create_engine(connection_string, echo=True)
    connection = engine.connect()
    meta_data = db.MetaData()

    # creation of Table for Training data sets for sql-database
    TrainingSets = db.Table(
        "Training_Sets", meta_data,
        db.Column("X", db.Float, nullable=False),
        db.Column("Y1", db.Float, nullable=False),
        db.Column("Y2", db.Float, nullable=False),
        db.Column("Y3", db.Float, nullable=False),
        db.Column("Y4", db.Float, nullable=False))

    # creation of Table for Ideal Functions data set for sql-database
    Ideal_functions = db.Table(
        "Ideal_functions", meta_data,
        db.Column("X", db.Float, nullable=False),
        db.Column("Y1", db.Float, nullable=False),
        db.Column("Y2", db.Float, nullable=False),
        db.Column("Y3", db.Float, nullable=False),
        db.Column("Y4", db.Float, nullable=False),
        db.Column("Y5", db.Float, nullable=False),
        db.Column("Y6", db.Float, nullable=False),
        db.Column("Y7", db.Float, nullable=False),
        db.Column("Y8", db.Float, nullable=False),
        db.Column("Y9", db.Float, nullable=False),
        db.Column("Y10", db.Float, nullable=False),
        db.Column("Y11", db.Float, nullable=False),
        db.Column("Y12", db.Float, nullable=False),
        db.Column("Y13", db.Float, nullable=False),
        db.Column("Y14", db.Float, nullable=False),
        db.Column("Y15", db.Float, nullable=False),
        db.Column("Y16", db.Float, nullable=False),
        db.Column("Y17", db.Float, nullable=False),
        db.Column("Y18", db.Float, nullable=False),
        db.Column("Y19", db.Float, nullable=False),
        db.Column("Y20", db.Float, nullable=False),
        db.Column("Y21", db.Float, nullable=False),
        db.Column("Y22", db.Float, nullable=False),
        db.Column("Y23", db.Float, nullable=False),
        db.Column("Y24", db.Float, nullable=False),
        db.Column("Y25", db.Float, nullable=False),
        db.Column("Y26", db.Float, nullable=False),
        db.Column("Y27", db.Float, nullable=False),
        db.Column("Y28", db.Float, nullable=False),
        db.Column("Y29", db.Float, nullable=False),
        db.Column("Y30", db.Float, nullable=False),
        db.Column("Y31", db.Float, nullable=False),
        db.Column("Y32", db.Float, nullable=False),
        db.Column("Y33", db.Float, nullable=False),
        db.Column("Y34", db.Float, nullable=False),
        db.Column("Y35", db.Float, nullable=False),
        db.Column("Y36", db.Float, nullable=False),
        db.Column("Y37", db.Float, nullable=False),
        db.Column("Y38", db.Float, nullable=False),
        db.Column("Y39", db.Float, nullable=False),
        db.Column("Y40", db.Float, nullable=False),
        db.Column("Y41", db.Float, nullable=False),
        db.Column("Y42", db.Float, nullable=False),
        db.Column("Y43", db.Float, nullable=False),
        db.Column("Y44", db.Float, nullable=False),
        db.Column("Y45", db.Float, nullable=False),
        db.Column("Y46", db.Float, nullable=False),
        db.Column("Y47", db.Float, nullable=False),
        db.Column("Y48", db.Float, nullable=False),
        db.Column("Y49", db.Float, nullable=False),
        db.Column("Y50", db.Float, nullable=False))

    # creation of Table for the found mapping
    Mapping = db.Table(
        "Mapping", meta_data,
        db.Column("X", db.Float, nullable=False),
        db.Column("Y", db.Float, nullable=False),
        db.Column("dY", db.String),
        db.Column("Ideal_function", db.String))

    # creation of all the tables in the sql-database
    meta_data.create_all(engine)

    # filling the sql tables for the training data
    training_table = db.Table("Training_Sets", meta_data, autoload=True,
                              autoload_with=engine)
    sql_query_training = db.insert(training_table)

    data_list_training = []
    row_training = {}

    for i in range(0, train_data.return_num_rows()):
        for u in range(0, train_data.return_num_cols()):

            if u == 0:
                row_training = {'X': train_data.return_xy_data(i, u)}
            else:
                row_training['Y' + str(u)] = train_data.return_xy_data(i, u)

        data_list_training.append(row_training)
        # reset
        row_training = {}

    result_training = connection.execute(sql_query_training, data_list_training)

    # filling the sql tables for the ideal functions
    ideal_func_table = db.Table("Ideal_functions", meta_data, autoload=True,
                                autoload_with=engine)
    sql_query_ideal_func = db.insert(ideal_func_table)

    data_list_ideal_func = []
    row_ideal = {}

    for i in range(0, ideal_data.return_num_rows()):
        for u in range(0, ideal_data.return_num_cols()):

            if u == 0:
                row_ideal = {'X': ideal_data.return_xy_data(i, u)}
            else:
                row_ideal['Y' + str(u)] = ideal_data.return_xy_data(i, u)

        data_list_ideal_func.append(row_ideal)
        # reset
        row_ideal = {}

    result_ideal_func = connection.execute(sql_query_ideal_func, data_list_ideal_func)

    # filling the sql tables for the mapping
    mapping_table = db.Table("Mapping", meta_data, autoload=True,
                             autoload_with=engine)
    sql_query_mapping = db.insert(mapping_table)

    data_list_mapping = []
    row_mapping = {}

    for i in range(0, test_data.return_num_rows()):
        for u in range(0, 5):

            if u == 0:
                row_mapping = {'X': test_data.return_xy_data(i, u)}
            elif u == 1:
                row_mapping['Y'] = test_data.return_xy_data(i, u)
            elif u == 2:
                if mapping[i] == None:
                    row_mapping['dY'] = "None"
                else:
                    row_mapping['dY'] = str(mapped_smallest_error[i])
            elif u == 3:
                if mapping[i] == None:
                    row_mapping['Ideal_function'] = "None"
                else:
                    row_mapping['Ideal_function'] = str(ideal_funcs_indices[mapping[i]])

        data_list_mapping.append(row_mapping)
        # reset
        row_mapping = {}

    result_mapping = connection.execute(sql_query_mapping, data_list_mapping)

create_database()


def extract_data(mapping, select_data):
    '''
    Function for filtering out the relevant x and y data of the test function that it can be
    used for visualization purposes, e.g. plotted in the determined ideal function.
    With the input "select_data", the corresponding relevant data will be returned:
    1: x and y data of the test data which matches to the first determined ideal function
    2: x and y data of the test data which matches to the second determined ideal function
    3: x and y data of the test data which matches to the third determined ideal function
    4: x and y data of the test data which matches to the fourth determined ideal function
    "None": x and y data that could not be assigned to any ideal function
    '''
    # variable initialization
    mapping = mapping
    matched_ideal4 = []
    matched_ideal3 = []
    matched_ideal2 = []
    matched_ideal1 = []
    matched_ideal_none = []
    x_values4 = []
    y_values4 = []
    x_values3 = []
    y_values3 = []
    x_values2 = []
    y_values2 = []
    x_values1 = []
    y_values1 = []
    x_values_none = []
    y_values_none = []

    # extract the relevant indices of the test data to the corresponding ideal function
    for i in range(0, len(mapping)):
        if mapping[i] == 3:
            matched_ideal4.append(i)
        elif mapping[i] == 2:
            matched_ideal3.append(i)
        elif mapping[i] == 1:
            matched_ideal2.append(i)
        elif mapping[i] == 0:
            matched_ideal1.append(i)
        else:
            matched_ideal_none.append(i)

            # adding the data to the corresponding list
    for i in matched_ideal4:
        x_values4.append(test_data.return_xy_data(i, 0))
        y_values4.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal3:
        x_values3.append(test_data.return_xy_data(i, 0))
        y_values3.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal2:
        x_values2.append(test_data.return_xy_data(i, 0))
        y_values2.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal1:
        x_values1.append(test_data.return_xy_data(i, 0))
        y_values1.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal_none:
        x_values_none.append(test_data.return_xy_data(i, 0))
        y_values_none.append(test_data.return_xy_data(i, 1))

    # returning the relevant data depending on the "select_data" input argument
    if select_data == 4:
        return (x_values4, y_values4)
    elif select_data == 3:
        return (x_values3, y_values3)
    elif select_data == 2:
        return (x_values2, y_values2)
    elif select_data == 1:
        return (x_values1, y_values1)
    elif select_data == "None":
        return (x_values_none, y_values_none)

# getting the test data assigned to the determined ideal function 4
(x_values4, y_values4) = extract_data(mapping, 4)
# getting the test data assigned to the determined ideal function 3
(x_values3, y_values3) = extract_data(mapping, 3)
# getting the test data assigned to the determined ideal function 2
(x_values2, y_values2) = extract_data(mapping, 2)
# getting the test data assigned to the determined ideal function 1
(x_values1, y_values1) = extract_data(mapping, 1)
# getting the test data not assigned to any ideal function
(x_values_none, y_values_none) = extract_data(mapping, "None")

# plot of ideal function found with first train data (row 1)
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[0]), label="ideal data")
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[0]) + (ideal_funcs_errors[0]*np.sqrt(2)), label="ideal data + error")
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[0]) - (ideal_funcs_errors[0]*np.sqrt(2)), label="ideal data - error")
plt.scatter(x_values1, y_values1, label="matching training points")
plt.legend()
plt.grid(True,color="k")
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title('First found ideal with boundaries')
plt.show()

# plot of ideal function found with second train data (row 2)
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[1]), label="ideal data")
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[1]) + (ideal_funcs_errors[1]*np.sqrt(2)), label="ideal data + error")
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[1]) - (ideal_funcs_errors[1]*np.sqrt(2)), label="ideal data - error")
plt.scatter(x_values2, y_values2, label="matching training points")
plt.legend()
plt.grid(True,color="k")
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title('Second found ideal with boundaries')
plt.show()

# plot of ideal function found with third train data (row 3)
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[2]), label="ideal data")
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[2]) + (ideal_funcs_errors[2]*np.sqrt(2)), label="ideal data + error")
plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[2]) - (ideal_funcs_errors[2]*np.sqrt(2)), label="ideal data - error")
plt.scatter(x_values3, y_values3, label="matching training points")
plt.legend()
plt.grid(True,color="k")
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title('Third found ideal with boundaries')
plt.show()