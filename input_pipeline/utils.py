"""File for loading different data"""
import pandas as pd
import logging
import numpy as np


class utils():
    def __init__(self, tabular_data_path, organ_columns_csv_path, label_csv_path, keys, data_type, two_label = False, only_same_age_range = False):
        self.tabular_data_path = tabular_data_path
        self.organ_columns_csv_path = organ_columns_csv_path
        self.keys = keys
        self.two_label = two_label
        self.only_same_age_range = only_same_age_range
        self.data_type = data_type
        self.first_assessment_age_labels = None
        self.image_assessment_age_labels = None
        self.label_csv_path = label_csv_path
        self.data = None
        self.age_labels = None

    def load_tabular_data(self, number_rows = 1000):
        """
        This method loads all the tabular data based on the given columns in the csv file.
        """
        logging.info("creating Tabular Data")

        def replace_column(target_column, source_column, df, index, reference_col):
            """
            This method fills NAN values from a column with coresponding values.
            """
            mask = merged_df[target_column].isna() & df[source_column].notna()
            df.loc[mask, target_column] = df.loc[mask, source_column]
            if reference_col is not None:
                reference_col[mask] = index
            return df, reference_col

        def update_columns(target_column, df, mapping, index, reference_col):
            """
            This method fills NAN values from a column with coresponding values.
            """
            for fill_col in mapping[target_column]:
                if int(fill_col[-3]) == index:
                    df, reference_col = replace_column(target_column, fill_col, df, index, reference_col)
            return df, reference_col


        headers = pd.read_csv(self.tabular_data_path, nrows=0).columns
        self.first_assessment_age_column = ['21003-0.0']
        self.image_assessment_age_columns = ['21003-2.0']

        # get the columns for the tabular data
        organ_columns_data = pd.read_csv(self.organ_columns_csv_path, dtype = {"Field-ID": str})
        if self.two_label == True:
            organ_columns_data = organ_columns_data[organ_columns_data["use-2-Label"] == 1]

        # get the mapping if multiple columns hold the same value
        organ_columns = pd.concat([pd.Series(["eid"]), organ_columns_data["Field-ID"]],ignore_index=True)
        mapping_corresponding_columns = organ_columns_data.loc[organ_columns_data["corresponding-column"].notna(), ["Field-ID", "corresponding-column"]]   
        mapping_corresponding_columns["corresponding-column"] = mapping_corresponding_columns["corresponding-column"].astype(int).astype(str)


        organ_columns = pd.concat([organ_columns, mapping_corresponding_columns["corresponding-column"]]).drop_duplicates().reset_index(drop=True)

        # get the desired columns from the header of the large UK Biobank csv
        if self.two_label == False and self.data_type == "multimodal":
            organ_headers = headers[headers.str.startswith(tuple(col + '-' for col in organ_columns)) & ~headers.str.endswith("-3.0") & headers.str.endswith(".0") | (headers == "eid")] # don't consider follow-up imaging (timepoint 3.0)
        else:
            organ_headers = headers[headers.str.startswith(tuple(col + '-' for col in organ_columns)) & headers.str.endswith("-2.0") | (headers == "eid")]  # CHANGE HERE

        headers_for_filling = organ_headers[organ_headers.str.match(r'^(' + '|'.join(mapping_corresponding_columns["corresponding-column"]) + r')-')]



        mapping_corresponding_columns["corresponding-column"] = mapping_corresponding_columns["corresponding-column"].apply(lambda col: organ_headers[organ_headers.str.startswith(col)].tolist())

        empty_list_mask = mapping_corresponding_columns["corresponding-column"].apply(lambda x: isinstance(x, list) and len(x) == 0)

        if empty_list_mask.any():
            missing_rows = mapping_corresponding_columns[empty_list_mask]
            field_ids = missing_rows["Field-ID"].tolist()
            raise ValueError(f"Missing correspondig column for Field-ID(s): {field_ids}. Please check that there is a -2.0 or -0.0 available of the correspondig column. For Training with two labels -0.0 has to be available")

        organ_headers = organ_headers.drop(headers_for_filling)

        if self.two_label == False and self.data_type == "multimodal":
            organ_headers_end_0 = organ_headers[organ_headers.str.endswith("-0.0")].str[:-4]
            organ_headers_end_1 = organ_headers[organ_headers.str.endswith("-1.0")].str[:-4]
            organ_headers_end_2 = organ_headers[organ_headers.str.endswith("-2.0")].str[:-4]

            # get all columns with suffix -2.0 that have a column with suffix -1.0 --> important for filling up
            organ_columns_2_with_1 = organ_headers[organ_headers.str.endswith("-2.0") & organ_headers.str[:-4].isin(organ_headers_end_1)]

            organ_columns_1_with_0_without_2 = organ_headers[organ_headers.str.endswith("-1.0") & organ_headers.str[:-4].isin(organ_headers_end_0)]
            organ_columns_1_with_0_without_2 = organ_columns_1_with_0_without_2[~organ_columns_1_with_0_without_2.isin(organ_headers[organ_headers.str.endswith("-1.0") & organ_headers.str[:-4].isin(organ_headers_end_2)])]

            organ_columns_only_2 = organ_headers[organ_headers.str.endswith("-2.0") & ~organ_headers.str[:-4].isin(organ_headers_end_1) & ~organ_headers.str[:-4].isin(organ_headers_end_0)]
            organ_columns_only_0 = organ_headers[organ_headers.str.endswith("-0.0") & ~organ_headers.str[:-4].isin(organ_headers_end_1) & ~organ_headers.str[:-4].isin(organ_headers_end_2)]

            deletable_columns = organ_headers[~organ_headers.isin(organ_columns_2_with_1.append(organ_columns_1_with_0_without_2).append(organ_columns_only_2).append(organ_columns_only_0).append(pd.Index(['eid'])))]

            organ_headers_cleaned = organ_headers.drop(list(deletable_columns))
            fillable_columns = organ_headers_cleaned.str.startswith(tuple(mapping_corresponding_columns["Field-ID"]))
            mapping_corresponding_columns["Field-ID"] = organ_headers_cleaned[fillable_columns]
            mapping_corresponding_columns = mapping_corresponding_columns.set_index("Field-ID")["corresponding-column"].to_dict()


            continous_columns = (organ_columns_data.loc[organ_columns_data["continous"] == 1, "Field-ID"])
            continous_columns = organ_headers_cleaned[organ_headers_cleaned.str.startswith(tuple(continous_columns))]

            organ_headers = organ_headers.append(headers_for_filling)

            deletable_columns = deletable_columns.append(headers_for_filling)
        
        else:
            fillable_columns = organ_headers.str.startswith(tuple(mapping_corresponding_columns["Field-ID"]))
            mapping_corresponding_columns["Field-ID"] = organ_headers[fillable_columns]
            mapping_corresponding_columns = mapping_corresponding_columns.set_index("Field-ID")["corresponding-column"].to_dict()

            continous_columns = organ_columns_data.loc[organ_columns_data["continous"] == 1, "Field-ID"]
            continous_columns = organ_headers[organ_headers.str.startswith(tuple(continous_columns))]

            organ_headers = organ_headers.append(headers_for_filling)
            deletable_columns = headers_for_filling



        try:
            organ_header_indexes = [headers.get_loc(col) for col in (list(organ_headers) + self.first_assessment_age_column + self.image_assessment_age_columns)]
        except KeyError as e:
            raise KeyError(f"Key {e} is not available") from e

        data = []
        len_df = 0
        len_keys = len(self.keys)
        columns_to_insert = {}
        reader = pd.read_csv(self.tabular_data_path, chunksize=number_rows, names=headers, skiprows=1, low_memory=False, usecols=organ_header_indexes)
        for idx, chunk in enumerate(reader):
            merged_df = chunk.merge(self.keys, on='eid', how='inner')
            if len_df >= len_keys:
                break
            
            if self.two_label == False and self.data_type == "multimodal":
                # extend the columns with reference columns
                for column in organ_columns_2_with_1:
                    reference_column_name = column[:-3] + '3.0'
                    reference_column = pd.Series(2, index=merged_df.index)

                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 2, reference_column)

                    Nan_mask = merged_df[column].isna() & merged_df[column[:-3] + '1.0'].notna()
                    merged_df.loc[Nan_mask, column] = merged_df.loc[Nan_mask, column[:-3] + '1.0']
                    reference_column[Nan_mask] = 1

                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 1, reference_column)

                    Nan_mask = merged_df[column].isna() & merged_df[column[:-3] + '0.0'].notna()
                    merged_df.loc[Nan_mask, column] = merged_df.loc[Nan_mask, column[:-3] + '0.0']
                    reference_column[Nan_mask] = 0

                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 0, reference_column)

                    Nan_mask = merged_df[column].isna()
                    reference_column[Nan_mask] = -1
                
                    columns_to_insert[reference_column_name] = reference_column

                for column in organ_columns_1_with_0_without_2:
                    reference_column_name = column[:-3] + '3.0'
                    reference_column = pd.Series(1, index=merged_df.index)

                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 1, reference_column)

                    Nan_mask = merged_df[column].isna() & merged_df[column[:-3] + '0.0'].notna()
                    merged_df.loc[Nan_mask, column] = merged_df.loc[Nan_mask, column[:-3] + '0.0']
                    reference_column[Nan_mask] = 0

                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 0, reference_column)

                    Nan_mask = merged_df[column].isna()
                    reference_column[Nan_mask] = -1

                    columns_to_insert[reference_column_name] = reference_column

                for column in organ_columns_only_2:
                    reference_column_name = column[:-3] + '3.0'
                    reference_column = pd.Series(2, index=merged_df.index)


                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 2, reference_column)
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 1, reference_column)
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 0, reference_column)

                    Nan_mask = merged_df[column].isna()
                    reference_column[Nan_mask] = -1

                    columns_to_insert[reference_column_name] = reference_column

                for column in organ_columns_only_0:
                    reference_column_name = column[:-3] + '3.0'
                    reference_column = pd.Series(0, index=merged_df.index)

                    if column in mapping_corresponding_columns:
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 2, reference_column)
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 1, reference_column)
                        merged_df, reference_column = update_columns(column, merged_df, mapping_corresponding_columns, 0, reference_column)

                    Nan_mask = merged_df[column].isna()
                    reference_column[Nan_mask] = -1

                    columns_to_insert[reference_column_name] = reference_column

                merged_df = pd.concat([merged_df, pd.DataFrame(columns_to_insert)], axis=1)
            
            else:
                for column in merged_df.columns:
                    if column in mapping_corresponding_columns:
                        merged_df, _ = update_columns(column, merged_df, mapping_corresponding_columns, 0, None)

            if self.only_same_age_range:
                merged_df = merged_df[merged_df["21003-0.0"] >= 44]

            merged_df = merged_df.drop(columns=deletable_columns)
            data.append(merged_df)
            len_df += len(merged_df)


        data = pd.concat(data, ignore_index = True)

        cols = data.columns.tolist()
        cols = sorted([col for col in cols])
        data = data[cols]

        nan_ratio = data.isna().mean()
        columns_with_10pct_nan = nan_ratio[nan_ratio >= 0.10].index.tolist()
        logging.info(f"Columns with less than 90% coverage: {columns_with_10pct_nan}")

        nan_ratio_rows = data.isna().mean(axis=1)
        if self.two_label == True or self.data_type == "tabular":
            limit = 0.3
        else:
            limit = 0.15
        deletable_eids = data.loc[nan_ratio_rows >= limit, 'eid'].tolist()
        if deletable_eids:
            data = data.loc[nan_ratio_rows < limit]
            logging.info(f"Dropped rows: {len(deletable_eids)}")


        deletable_eids_first_assessment = data.loc[data[self.first_assessment_age_column].isna().any(axis=1), "eid"].tolist()
        if deletable_eids_first_assessment:
            logging.info(f"Dropped rows because age at first assessment is None with EIDs: {deletable_eids_first_assessment}")
        data = data.dropna(subset=self.first_assessment_age_column)
        data = data.fillna(0)

        data[continous_columns] = (data[continous_columns] - data[continous_columns].mean()) / data[continous_columns].std()


        self.keys = data[['eid']].copy() 

        self.first_assessment_age_labels = data[self.first_assessment_age_column]
        self.age_labels = data[['eid'] + self.first_assessment_age_column + self.image_assessment_age_columns]
        data = data.drop(columns=self.first_assessment_age_column + self.image_assessment_age_columns + ['eid'])

        FLOAT16_MAX = np.finfo(np.float16).max
        FLOAT16_MIN = -FLOAT16_MAX
        for col in data.columns:
            col_max = data[col].max()
            col_min = data[col].min()
            if col_max > FLOAT16_MAX:
                logging.info(f"Column '{col}' has values out of float16 range: max={col_max}")
                data[col] = data[col].clip(upper=FLOAT16_MAX)
            if col_min < FLOAT16_MIN:
                logging.info(f"Column '{col}' has values out of float16 range: min={col_min}")
                data[col] = data[col].clip(lower=FLOAT16_MIN)

        logging.info("Tabular Data created")

        self.data = data

        return data, self.keys

    def load_age_labels(self):
        """
        This method loads the labels.
        """
        """labels = pd.read_csv(self.label_csv_path, usecols = [1,2], names = ['eid', 'age'], skiprows = 1)
        merged_df = labels.merge(self.keys, on='eid', how='inner')
        merged_df = merged_df.dropna(subset=['age'])
        if self.only_same_age_range:
            merged_df = merged_df[merged_df["age"] <= 70]
        self.keys = merged_df[["eid"]]
        merged_df = merged_df.drop(columns = ['eid'])
        return merged_df, self.keys"""
        if self.age_labels is not None:
            labels = self.age_labels
            labels = labels.rename(columns={self.image_assessment_age_columns[0]: 'age'})   # CHANGE HERE
            labels = labels.dropna(subset=['age'])
            self.keys = labels[['eid']].copy()
            labels = labels[['age']].copy()
            return labels, self.keys
        else:
            labels = pd.read_csv(self.label_csv_path, usecols = [1,2], names = ['eid', 'age'], skiprows = 1)
            merged_df = labels.merge(self.keys, on='eid', how='inner')
            merged_df = merged_df.dropna(subset=['age'])
            if self.only_same_age_range:
                merged_df = merged_df[merged_df["age"] <= 70]
            self.keys = merged_df[["eid"]]
            merged_df = merged_df.drop(columns = ['eid'])
            return merged_df, self.keys

    def get_first_assessment_age_labels(self):
        """
        This method returns the CA at the first assessment.
        """
        if self.first_assessment_age_labels is None:
            raise Exception(f"Please load data first")
        return self.first_assessment_age_labels

    def get_tabular_size(self):
        """
        This method returns the amount of columns in the tabular data.
        """
        if self.data is None:
            raise Exception(f"Please Load Data before trying to access its size")
        else:
            return self.data.shape[1]



