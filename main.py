# Homework #1: Relational Algebra Implementation using Pandas DataFrames
# Includes: Basic Operators, Additional Operators, CLI, File Loading,
#           Error Handling, Bug Fixes, and 'source' command.

import pandas as pd
import shlex  # For robust command line splitting
import sys
import os
from io import StringIO # To potentially load data from strings for testing
import traceback # For potentially logging detailed errors if needed

# --- Global dictionary to store tables (DataFrames) ---
tables = {}

# --- Helper Functions ---

def _check_tables_exist(table_names, operation_name):
    """Checks if all table names exist in the global 'tables' dict."""
    # Ensure table_names is a list or tuple
    if isinstance(table_names, str):
        table_names = [table_names]
    missing = [name for name in table_names if name not in tables]
    if missing:
        print(f"Error ({operation_name}): Table(s) not found: {', '.join(missing)}")
        return False
    return True

def _check_schema_compatibility(df1, df2, operation_name):
    """Checks if two dataframes have the same column names and order."""
    if list(df1.columns) != list(df2.columns):
        print(f"Error ({operation_name}): Schemas are incompatible.")
        print(f"  Schema 1: {list(df1.columns)}")
        print(f"  Schema 2: {list(df2.columns)}")
        return False
    return True

def _parse_columns(col_str):
    """Parses a comma-separated string of column names."""
    # Strip whitespace around commas and from column names
    return [col.strip() for col in col_str.split(',') if col.strip()]


def _parse_rename_map(map_str):
    """Parses a comma-separated string of 'old:new' pairs into a dict."""
    rename_map = {}
    try:
        # Allow spaces around comma, but not within names unless quoted (handled by shlex later)
        pairs = [p.strip() for p in map_str.split(',')]
        for pair in pairs:
            if ':' not in pair:
                raise ValueError(f"Invalid rename pair (missing ':'): {pair}")
            old_name, new_name = pair.split(':', 1)
            # Strip potential whitespace again from individual names
            old_name_stripped = old_name.strip()
            new_name_stripped = new_name.strip()
            if not old_name_stripped or not new_name_stripped:
                 raise ValueError(f"Invalid rename pair (empty name): {pair}")
            rename_map[old_name_stripped] = new_name_stripped
        return rename_map
    except Exception as e:
        print(f"Error parsing rename map '{map_str}': {e}")
        return None

# --- 1. Basic Operators (Six Functions) ---

def select(df, condition_str):
    """
    Performs the SELECT operation (σ).
    Filters rows based on a condition string (Pandas query format).
    """
    if df.empty and condition_str:
         # Return the empty dataframe if input is empty
         return df.copy()
    try:
        result_df = df.query(condition_str).reset_index(drop=True)
        return result_df
    except Exception as e:
        print(f"Error during SELECT operation: {e}")
        print("  Ensure your condition uses valid column names and syntax.")
        print('  Example: "Age > 20 and Major == \'CS\'" (note quotes for strings)')
        return None # Return None on failure

def project(df, columns):
    """
    Performs the PROJECT operation (π).
    Selects specific columns from the dataframe.
    'columns' is a list of column names. Assumes unique column names passed in.
    """
    if df.empty:
         # If input is empty, return an empty DF with the requested columns
         return pd.DataFrame(columns=columns)
    try:
        # Check if all requested columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Column(s) not found: {', '.join(missing_cols)}")

        # Use drop_duplicates to mimic set semantics of projection in RA
        # Select columns first, then drop duplicates
        result_df = df[columns].drop_duplicates().reset_index(drop=True)
        return result_df
    except Exception as e:
        print(f"Error during PROJECT operation: {e}")
        return None

def rename(df, rename_map):
    """
    Performs the RENAME operation (ρ).
    Renames columns based on the provided map {old_name: new_name}.
    """
    if df.empty:
         # If input is empty, rename an empty DF (will have new column names)
         empty_renamed = pd.DataFrame(columns=df.columns).rename(columns=rename_map)
         return empty_renamed
    try:
        # Check if all old names exist in the DataFrame columns
        missing_keys = [old for old in rename_map.keys() if old not in df.columns]
        if missing_keys:
            raise ValueError(f"Column(s) to rename not found: {', '.join(missing_keys)}")
        # Check if any new name conflicts with existing columns not being renamed
        existing_other_cols = set(df.columns) - set(rename_map.keys())
        new_name_conflicts = [new for new in rename_map.values() if new in existing_other_cols]
        if new_name_conflicts:
             raise ValueError(f"New name(s) conflict with existing columns: {', '.join(new_name_conflicts)}")
        # Check for conflicts among new names themselves if multiple old map to same new
        if len(rename_map.values()) != len(set(rename_map.values())):
            from collections import Counter
            counts = Counter(rename_map.values())
            duplicate_new = [name for name, count in counts.items() if count > 1]
            raise ValueError(f"Multiple columns renamed to the same new name: {', '.join(duplicate_new)}")


        result_df = df.rename(columns=rename_map)
        return result_df
    except Exception as e:
        print(f"Error during RENAME operation: {e}")
        return None

def cartesian_product(df1_orig, df2_orig):
    """
    Performs the CARTESIAN PRODUCT operation (×).
    Works on copies of inputs to avoid side effects.
    Handles potential column name collisions by adding suffixes (_x, _y).
    """
    # Handle empty inputs
    if df1_orig.empty or df2_orig.empty:
        # The cartesian product involving an empty set is empty, but need the combined schema
        cols1 = list(df1_orig.columns)
        cols2 = list(df2_orig.columns)
        # Handle potential name clashes in the empty result schema
        final_cols = []
        cols2_set = set(cols2)
        for c1 in cols1:
            if c1 in cols2_set:
                final_cols.append(f"{c1}_x")
            else:
                final_cols.append(c1)
        for c2 in cols2:
            if c2 in cols1:
                 final_cols.append(f"{c2}_y")
            else:
                 final_cols.append(c2)
        return pd.DataFrame(columns=final_cols)


    # Work with copies to prevent modifying the original DataFrames in the 'tables' dict
    df1 = df1_orig.copy()
    df2 = df2_orig.copy()
    temp_key = '_temp_cross_join_key' # Use a name unlikely to clash

    # Ensure the temp_key is unique if somehow present
    while temp_key in df1.columns or temp_key in df2.columns:
        temp_key += "_"

    try:
        df1[temp_key] = 1
        df2[temp_key] = 1

        # Perform the merge (cross join). Suffixes handle overlaps.
        result_df = pd.merge(df1, df2, on=temp_key, suffixes=('_x', '_y'))

        # Remove the temporary key from the result
        result_df = result_df.drop(temp_key, axis=1)

        # No need to drop the key from df1 and df2 as they are local copies

        return result_df.reset_index(drop=True)

    except Exception as e:
        print(f"Error during CARTESIAN PRODUCT operation: {e}")
        # No cleanup needed on df1, df2 as they are local copies
        return None

def union(df1, df2):
    """
    Performs the SET UNION operation (∪).
    Combines rows from two dataframes, removing duplicates.
    Requires schemas to be compatible (same column names and order).
    """
    if not _check_schema_compatibility(df1, df2, "UNION"):
        return None
    # Handle cases where one or both are empty
    if df1.empty and df2.empty:
         return df1.copy() # Return empty frame with correct schema
    if df1.empty:
         return df2.drop_duplicates().reset_index(drop=True)
    if df2.empty:
         return df1.drop_duplicates().reset_index(drop=True)

    try:
        # ignore_index=True creates a new index
        # drop_duplicates removes identical rows
        result_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)
        return result_df
    except Exception as e:
        print(f"Error during UNION operation: {e}")
        return None

def difference(df1, df2):
    """
    Performs the SET DIFFERENCE operation (-). R - S.
    Returns rows present in df1 but not in df2.
    Requires schemas to be compatible (same column names and order).
    """
    if not _check_schema_compatibility(df1, df2, "DIFFERENCE"):
        return None
    # Handle empty inputs
    if df1.empty:
        return df1.copy() # R is empty, so R-S is empty
    if df2.empty:
        return df1.drop_duplicates().reset_index(drop=True) # S is empty, so R-S = R (unique)

    try:
        # Use merge with indicator=True. Rows only in df1 will have _merge == 'left_only'.
        # Use all columns as the key for matching rows exactly.
        merge_cols = list(df1.columns)
        if not merge_cols: # Handle case of dataframes with no columns (unlikely but possible)
             return df1.copy() if df2.empty else pd.DataFrame() # Difference depends if S is also no-column empty

        merged = df1.merge(df2.drop_duplicates(), how='left', indicator=True, on=merge_cols)
        result_df = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

        # Ensure result has same columns as df1 (it should by definition of merge)
        result_df = result_df[df1.columns]

        # Drop duplicates from the result itself (if df1 had duplicates not in df2)
        return result_df.drop_duplicates().reset_index(drop=True)

    except Exception as e:
        print(f"Error during DIFFERENCE operation: {e}")
        return None

# --- 2. Additional Operators (Calling the Six Functions) ---

def intersection(df1, df2):
    """
    Performs the SET INTERSECTION operation (∩). R ∩ S.
    Returns rows present in both df1 and df2.
    Implemented as: R - (R - S)
    Requires schemas to be compatible.
    """
    print("  Executing Intersection (R ∩ S) using R - (R - S)")
    if not _check_schema_compatibility(df1, df2, "INTERSECTION"):
         return None # Schema check needed early

    # Handle empty inputs
    if df1.empty or df2.empty:
        return pd.DataFrame(columns=df1.columns) # Intersection with empty set is empty

    print("    Step 1: Calculating R - S")
    r_minus_s = difference(df1, df2)
    if r_minus_s is None:
        print("    Intersection failed: Step 1 (R - S) failed.")
        return None
    print(f"      Intermediate result (R - S) has {len(r_minus_s)} rows.")

    print("    Step 2: Calculating R - (R - S)")
    result_df = difference(df1, r_minus_s)
    if result_df is None:
         print("    Intersection failed: Step 2 (R - (R-S)) failed.")
         return None

    print("  Intersection completed.")
    # The result of difference is already unique
    return result_df

def natural_join(df1_orig, df2_orig):
    """
    Performs the NATURAL JOIN operation (R ⨝ S).
    Combines rows based on matching values in common columns.
    Implemented using cartesian product, select, project, rename.
    Works on copies to avoid side effects.
    """
    print("  Executing Natural Join (R ⨝ S) using basic operators")

    # Use copies to avoid modifying original tables passed to the function
    df1 = df1_orig.copy()
    df2 = df2_orig.copy()

    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    common_cols = sorted([col for col in cols1 if col in cols2]) # Use sorted list for predictable order
    unique_cols1 = sorted([col for col in cols1 if col not in common_cols])
    unique_cols2 = sorted([col for col in cols2 if col not in common_cols])

    final_column_order = common_cols + unique_cols1 + unique_cols2

    # Handle cases with empty inputs
    if df1.empty or df2.empty:
        print("    Natural join with an empty table results in an empty table.")
        return pd.DataFrame(columns=final_column_order)

    if not common_cols:
        print("    Warning: No common columns found. Natural join defaults to Cartesian Product.")
        # Ensure cartesian product is called with original copies if needed, but here df1, df2 are already copies
        return cartesian_product(df1, df2)

    print(f"    Common columns: {common_cols}")
    print(f"    Unique to R: {unique_cols1}")
    print(f"    Unique to S: {unique_cols2}")

    # --- Strategy: ---
    # 1. Cartesian Product: R x S. Pandas merge will add suffixes (_x, _y) for *all* overlapping columns.
    # 2. Select: Filter rows where common_col_x == common_col_y for all common columns.
    # 3. Project: Keep one version of common columns (from _x), all unique columns from R (handle _x), unique from S (handle _y).
    # 4. Rename: Remove suffixes (_x, _y) to get the final schema.

    print("    Step 1: Calculating Cartesian Product (R × S)")
    cp = cartesian_product(df1, df2) # df1, df2 are already copies
    if cp is None:
        print("    Natural Join failed: Step 1 (Cartesian Product) failed.")
        return None
    if cp.empty: # If CP is empty (can happen if one input was empty, handled above, but check anyway)
         print("    Natural join result is empty (Cartesian Product was empty).")
         return pd.DataFrame(columns=final_column_order)
    print(f"      CP result columns: {list(cp.columns)}")


    # --- Step 2: Build Selection Condition ---
    select_conditions = []
    for col in common_cols:
        # Pandas merge used in cartesian_product adds _x and _y for ALL overlapping columns
        col_x = f"{col}_x"
        col_y = f"{col}_y"
        if col_x in cp.columns and col_y in cp.columns:
             # Use backticks for safety with potential special characters in names
             select_conditions.append(f"`{col_x}` == `{col_y}`")
        else:
            print(f"    Error: Expected columns '{col_x}' and '{col_y}' not found in Cartesian Product result for common column '{col}'.")
            print(f"    Natural Join failed: Step 2 failed finding join columns for '{col}' in CP result.")
            return None

    if not select_conditions: # Should have conditions if common_cols is not empty
        print("    Natural Join failed: Step 2 could not form any join conditions (Internal Error).")
        return None

    condition_str = " and ".join(select_conditions)
    print(f"    Step 3: Selecting rows based on condition: {condition_str}")
    selected_cp = select(cp, condition_str)
    if selected_cp is None:
        print("    Natural Join failed: Step 3 (Select) failed.")
        return None
    if selected_cp.empty:
         print("    Natural join result is empty after selection.")
         return pd.DataFrame(columns=final_column_order)

    # --- Step 4: Determine Columns to Project and Rename Map ---
    project_cols_intermediate = []
    final_rename_map = {}

    # Add common columns (take the '_x' version) and set up rename
    for col in common_cols:
        col_x = f"{col}_x"
        if col_x in selected_cp.columns:
            project_cols_intermediate.append(col_x)
            final_rename_map[col_x] = col # Rename col_x back to col
        else: # This should not happen if select worked
             print(f"    Error: Column '{col_x}' expected but not found after select (Internal Error).")
             return None

    # Add unique columns from R (check if they got _x suffix)
    for col in unique_cols1:
        col_x = f"{col}_x"
        if col_x in selected_cp.columns: # Got suffix because it overlapped with a df2 column
             project_cols_intermediate.append(col_x)
             final_rename_map[col_x] = col
        elif col in selected_cp.columns: # Did not get suffix
             project_cols_intermediate.append(col)
        else: # Should not happen
             print(f"    Error: Column '{col}' or '{col_x}' expected but not found after select (Internal Error).")
             return None


    # Add unique columns from S (check if they got _y suffix)
    for col in unique_cols2:
        col_y = f"{col}_y"
        if col_y in selected_cp.columns: # Got suffix (expected)
            project_cols_intermediate.append(col_y)
            final_rename_map[col_y] = col
        elif col in selected_cp.columns: # Did not get suffix (unlikely with standard merge)
             print(f"    Warning: Column '{col}' from S did not get expected suffix '_y'. Using direct name.")
             project_cols_intermediate.append(col)
        else: # Should not happen
             print(f"    Error: Column '{col}' or '{col_y}' expected but not found after select (Internal Error).")
             return None

    # Ensure the intermediate project list only contains unique column names before calling project
    unique_project_cols_intermediate = []
    seen_proj_cols = set()
    for p_col in project_cols_intermediate:
        if p_col not in seen_proj_cols:
            unique_project_cols_intermediate.append(p_col)
            seen_proj_cols.add(p_col)

    print(f"    Step 4: Projecting intermediate columns: {unique_project_cols_intermediate}")

    if not unique_project_cols_intermediate: # Should not happen if we got this far
        print("    Natural Join failed: No columns left to project in Step 4 (Internal Error).")
        return pd.DataFrame(columns=final_column_order)

    # Core project function expects unique columns
    projected_result = project(selected_cp, unique_project_cols_intermediate)
    if projected_result is None:
        print("    Natural Join failed: Step 4 (Project) failed.")
        return None

    # Filter the rename map based on columns actually present after projection
    final_rename_map_filtered = {k: v for k, v in final_rename_map.items() if k in projected_result.columns}

    # --- Step 5: Final Rename ---
    if final_rename_map_filtered:
        print(f"    Step 5: Renaming columns to final names: {final_rename_map_filtered}")
        final_result = rename(projected_result, final_rename_map_filtered)
        if final_result is None:
            print("    Natural Join failed: Step 5 (Rename) failed.")
            return None
    else:
        final_result = projected_result # No rename needed

    # Reorder columns: common, unique1, unique2 for consistency
    # Ensure the final result only includes columns that should be there and exist
    final_result = final_result[[col for col in final_column_order if col in final_result.columns]]

    print("  Natural Join completed.")
    # Result of project should already be unique rows
    return final_result.reset_index(drop=True)


def division(df1_orig, df2_orig):
    """
    Performs the DIVISION operation (R ÷ S). R(A, B), S(B). Result(A).
    Finds tuples in R (projected onto A) such that their combinations
    with *all* tuples in S exist in R.
    Implemented using: π_A(R) - π_A((π_A(R) × S) - R)
    Works on copies to avoid side effects.
    """
    print("  Executing Division (R ÷ S) using basic operators")

    # Use copies to avoid modifying original tables
    R = df1_orig.copy()
    S = df2_orig.copy()

    cols_R = set(R.columns)
    cols_S = set(S.columns)

    attrs_A = sorted(list(cols_R - cols_S))
    attrs_B = sorted(list(cols_S)) # Common attributes
    final_schema = attrs_A # Schema of the result

    if not cols_S:
        print(f"Error (DIVISION): Divisor table S cannot have an empty schema (no columns).")
        return None
    if not cols_S.issubset(cols_R):
        print(f"Error (DIVISION): Schema of S ({attrs_B}) must be a subset of the schema of R ({sorted(list(cols_R))}).")
        return None
    if not attrs_A:
        print("Error (DIVISION): No attributes unique to R (attributes A). Division requires attributes A.")
        # Result should be an empty table with no columns or potentially a single boolean value depending on definition.
        # Return empty table with no columns for simplicity.
        return pd.DataFrame()

    # Handle empty inputs
    if R.empty:
        print("    Division with empty dividend R results in an empty table.")
        return pd.DataFrame(columns=final_schema)
    if S.empty:
         print(f"Error (DIVISION): Division by an empty table S is undefined in standard relational algebra.")
         print(f"  (Returning empty table with schema {final_schema}.)")
         return pd.DataFrame(columns=final_schema)

    print(f"    Attributes A (result schema): {attrs_A}")
    print(f"    Attributes B (matching schema): {attrs_B}")

    # --- Step 1: π_A(R) ---
    print("    Step 1: Calculating π_A(R)")
    pi_A_R = project(R, attrs_A)
    if pi_A_R is None:
        print("    Division failed: Step 1 (Project A from R) failed.")
        return None
    if pi_A_R.empty: # Already unique from project
         print("    Result of π_A(R) is empty. Division result is empty.")
         return pi_A_R # Return the empty dataframe with correct columns A

    # --- Step 2: π_A(R) × S ---
    print("    Step 2: Calculating π_A(R) × S")
    pi_A_R_x_S = cartesian_product(pi_A_R, S) # Uses copies internally
    if pi_A_R_x_S is None:
        print("    Division failed: Step 2 (Cartesian Product) failed.")
        return None
    if pi_A_R_x_S.empty:
        # This could happen if pi_A_R or S was empty, handled above, but check defensively
        print("    Intermediate cartesian product π_A(R) × S is empty.")
        # If this is empty, the difference in Step 3 will be empty, pi_A of that is empty,
        # so final result is pi_A(R) - empty = pi_A(R). Let's continue.

    # --- Standardize columns of CP result to match R's schema (A+B) ---
    expected_cols_R_AB = attrs_A + attrs_B
    actual_cp_cols = list(pi_A_R_x_S.columns)
    temp_rename_map_cp = {}
    project_cols_cp = []
    valid_cp_schema = True
    try:
        for col in expected_cols_R_AB:
            found = False
            # Check for original name, _x suffixed name, _y suffixed name
            possible_names = [col, f"{col}_x", f"{col}_y"]
            for name in possible_names:
                 if name in actual_cp_cols:
                     if name not in project_cols_cp: # Avoid adding duplicates if name collision was weird
                         project_cols_cp.append(name)
                         if name != col:
                             temp_rename_map_cp[name] = col
                     found = True
                     break
            if not found:
                 print(f"    Error: Expected column related to '{col}' not found in CP result columns: {actual_cp_cols}.")
                 valid_cp_schema = False
                 break
        if not valid_cp_schema: raise ValueError("Column mismatch")

        # Create the dataframe with standardized names for the difference
        # Project first, then rename
        pi_A_R_x_S_proj = project(pi_A_R_x_S, project_cols_cp) # project ensures uniqueness if needed
        if pi_A_R_x_S_proj is None: raise ValueError("Projection failed")

        if temp_rename_map_cp:
            pi_A_R_x_S_std = rename(pi_A_R_x_S_proj, temp_rename_map_cp)
            if pi_A_R_x_S_std is None: raise ValueError("Rename failed")
        else:
            pi_A_R_x_S_std = pi_A_R_x_S_proj

        # Ensure column order matches R for difference
        pi_A_R_x_S_std = pi_A_R_x_S_std[expected_cols_R_AB]

    except Exception as e:
         print(f"    Division failed: Error standardizing columns after Cartesian Product in Step 2: {e}")
         return None


    # --- Step 3: (π_A(R) × S) - R ---
    print(f"    Step 3: Calculating (π_A(R) × S) - R")
    # Project R to the same schema A+B first for the difference
    R_AB = project(R, expected_cols_R_AB) # project ensures unique rows
    if R_AB is None:
         print("    Division failed: Step 3a (Project R to A, B) failed.")
         return None
    R_AB = R_AB[expected_cols_R_AB] # Ensure order

    # Check schema compatibility before difference (should match by construction now)
    if not _check_schema_compatibility(pi_A_R_x_S_std, R_AB, "DIVISION Step 3"):
         print(f"    Division failed: Schema mismatch before difference in step 3 (Internal Error).")
         return None

    missing_tuples = difference(pi_A_R_x_S_std, R_AB) # Result is unique
    if missing_tuples is None:
        print("    Division failed: Step 3b (Difference) failed.")
        return None

    # --- Step 4: π_A((π_A(R) × S) - R) ---
    print(f"    Step 4: Calculating π_A of the 'missing tuples' result")
    if missing_tuples.empty:
        pi_A_missing = pd.DataFrame(columns=attrs_A) # Empty frame with correct schema A
        print("      (Difference result in Step 3 was empty)")
    else:
        pi_A_missing = project(missing_tuples, attrs_A) # Project ensures unique A's
        if pi_A_missing is None:
            print("    Division failed: Step 4 (Project A from Difference) failed.")
            return None

    # --- Step 5: π_A(R) - π_A(...) ---
    print("    Step 5: Calculating final difference: π_A(R) - (result from Step 4)")
    # Check schema compatibility (should match: attrs_A)
    if not _check_schema_compatibility(pi_A_R, pi_A_missing, "DIVISION Step 5"):
         print(f"    Division failed: Schema mismatch before final difference in step 5 (Internal Error).")
         return None

    result_df = difference(pi_A_R, pi_A_missing) # Result is unique
    if result_df is None:
        print("    Division failed: Step 5 (Final Difference) failed.")
        return None

    print("  Division completed.")
    return result_df.reset_index(drop=True)


# --- 3. User Interface (CLI) ---

def handle_load(args):
    """
    Loads data from a CSV file into a table.
    If the file is not found, creates an empty table with that name.
    """
    # 1. Check Argument Count and Format
    usage = "Usage: load <filename.csv> <table_name>"
    if len(args) != 2:
        print(usage)
        print("Error: Incorrect number of arguments.")
        if '#' in args:
            print("       (Ensure comments '#' are on separate lines or removed from the command)")
        return
    filename, table_name = args

    # 2. Validate Table Name
    if not table_name.isidentifier():
         print(f"Error: Invalid table name '{table_name}'. Use letters, numbers, and underscores, not starting with a number.")
         return

    try:
        # 3. Check if File Exists
        if not os.path.exists(filename):
            # --- File Not Found: Create Empty Table ---
            print(f"Warning: File '{filename}' not found.")
            # Decide whether to create the physical file (optional)
            create_physical_file = False # Set to True to also create empty file
            if create_physical_file:
                try:
                    with open(filename, 'w') as f: pass
                    print(f"   Created empty file '{filename}' on disk.")
                except Exception as e_create:
                    print(f"   Warning: Could not create empty file '{filename}' on disk: {e_create}")

            # Create an empty DataFrame in memory
            tables[table_name] = pd.DataFrame()
            print(f"Created empty table '{table_name}' in memory.")
            return # Successfully "loaded" an empty table

        # --- File Exists: Proceed with Loading ---
        # 4. Read CSV
        df = pd.read_csv(filename, float_precision='round_trip', low_memory=False)

        # 5. Handle Case Where File Exists but is Empty/No Header
        if df.empty and df.columns.empty:
             try:
                 file_size = os.path.getsize(filename)
                 if file_size == 0: print(f"Warning: File '{filename}' exists but is completely empty. Loaded as an empty table '{table_name}'.")
                 else: print(f"Warning: File '{filename}' exists but contains no parsable data or header. Loaded as an empty table '{table_name}'.")
             except OSError: print(f"Warning: File '{filename}' may be empty or unreadable. Loaded as an empty table '{table_name}'.")
             tables[table_name] = pd.DataFrame()
             return

        # 6. Sanitize Column Names
        original_cols = list(df.columns)
        df.columns = ["_".join(str(col).split()) for col in df.columns]
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
        renamed_cols_map = {}
        final_cols = []
        for i, col in enumerate(df.columns):
            original_name = original_cols[i]
            new_col = col
            if not new_col: new_col = f"column_{i}"
            elif new_col[0].isdigit(): new_col = f"_{new_col}"
            col_count = 0
            base_col = new_col
            while new_col in final_cols:
                col_count += 1; new_col = f"{base_col}_{col_count}"
            if new_col != original_name: renamed_cols_map[str(original_name)] = new_col
            final_cols.append(new_col)
        df.columns = final_cols

        # 7. Store DataFrame and Report Success
        tables[table_name] = df
        print(f"Table '{table_name}' loaded from '{filename}' with {len(df)} rows.")
        print(f"  Columns: {list(df.columns)}")
        if renamed_cols_map:
             renamed_msgs = [f"'{o}'->'{n}'" for o, n in renamed_cols_map.items()]
             print(f"  Note: Some columns were renamed for compatibility: {', '.join(renamed_msgs)}")

    # 8. General Error Handling during Load
    except pd.errors.EmptyDataError:
        print(f"Warning: File '{filename}' contains no data or header. Loaded as an empty table '{table_name}'.")
        tables[table_name] = pd.DataFrame()
        return
    except Exception as e:
        print(f"Error during load operation for '{filename}': {e}")
        if table_name in tables:
            try: del tables[table_name]
            except KeyError: pass


def handle_show(args):
    """Shows schema of loaded tables or contents of a specific table."""
    if not args:
        # Show all loaded table names
        if not tables: print("No tables loaded.")
        else:
            print("Loaded tables:")
            for name in sorted(tables.keys()):
                df = tables[name]
                print(f"- {name} ({len(df)} rows, {len(df.columns)} columns)")
    elif len(args) == 1:
         target = args[0]
         if target == 'schema':
              print("Usage: show schema <table_name>\nOr:    show <table_name>\nOr:    show")
         elif target in tables:
             print(f"Table '{target}':")
             if tables[target].empty: print(tables[target])
             else:
                 pd.options.display.max_rows = 100 # Limit printed rows
                 print(tables[target].to_string())
         else: print(f"Error: Table '{target}' not found.")
    elif len(args) == 2 and args[0] == 'schema':
        table_name = args[1]
        if table_name in tables:
            print(f"Schema for table '{table_name}':")
            buf = StringIO(); tables[table_name].info(buf=buf)
            print(buf.getvalue())
        else: print(f"Error: Table '{table_name}' not found.")
    else:
        print("Invalid 'show' command.\nUsage: show schema <table_name>\n       show <table_name>\n       show")


def handle_quit(args):
    """Exits the program."""
    if args: print("Warning: 'quit' command does not take any arguments. Ignoring extra input.")
    print("Quitting Relational Algebra CLI.")
    sys.exit(0)


def _generic_binary_handler(args, command_name, func, operation_symbol):
    """Helper for binary operators like union, diff, intersect, join, cp, divide."""
    usage = f"Usage: {command_name} <table1> <table2> <output_table>"
    example = f"Example: {command_name} R S result_table"
    if len(args) != 3:
        print(usage); print(example); print("Error: Incorrect number of arguments.")
        if '#' in args: print("       (Ensure comments '#' are on separate lines or removed from the command)")
        return
    t1_name, t2_name, out_table = args

    if not out_table.isidentifier():
         print(f"Error: Invalid output table name '{out_table}'. Use letters, numbers, and underscores, not starting with a number.")
         return
    if out_table in [t1_name, t2_name]:
        print(f"Warning: Output table name '{out_table}' is the same as an input table. The input table will be overwritten.")

    if not _check_tables_exist([t1_name, t2_name], command_name.upper()): return

    print(f"Executing: {out_table} = {t1_name} {operation_symbol} {t2_name}")
    # Ensure inputs exist before calling the function (redundant check, but safe)
    if t1_name not in tables or t2_name not in tables:
        print(f"Error ({command_name.upper()}): Input table(s) disappeared unexpectedly.")
        return

    result = func(tables[t1_name], tables[t2_name]) # Call the specific RA function

    if result is not None:
        tables[out_table] = result
        print(f"{command_name.upper()} operation complete. Result saved as '{out_table}' ({len(result)} rows).")
        if result.empty: print(result)
        else:
             rows_to_show = min(len(result), 10)
             print(f"First {rows_to_show} rows of '{out_table}':")
             print(result.head(rows_to_show).to_string())
    else:
        print(f"{command_name.upper()} operation failed.")

# --- Command Handlers using the generic helper where applicable ---

def handle_select(args):
    """Handles the SELECT command."""
    usage = "Usage: select <input_table> \"<condition_string>\" <output_table>"
    example = 'Example: select students "Age > 20 and Major == \'CS\'" selected_students'
    if len(args) != 3:
        print(usage); print(example); print("Error: Incorrect number of arguments.")
        if '#' in args: print("       (Ensure comments '#' are on separate lines or removed from the command)")
        return
    in_table, condition, out_table = args

    if not out_table.isidentifier():
         print(f"Error: Invalid output table name '{out_table}'. Use letters, numbers, and underscores, not starting with a number.")
         return
    if out_table == in_table:
         print(f"Warning: Output table name '{out_table}' is the same as the input table. The input table will be overwritten.")

    if not _check_tables_exist([in_table], "SELECT"): return

    print(f"Executing: {out_table} = σ ({condition}) ({in_table})")
    result = select(tables[in_table], condition)
    if result is not None:
        tables[out_table] = result
        print(f"SELECT operation complete. Result saved as '{out_table}' ({len(result)} rows).")
        if result.empty: print(result)
        else:
             rows_to_show = min(len(result), 10)
             print(f"First {rows_to_show} rows of '{out_table}':")
             print(result.head(rows_to_show).to_string())
    else:
        print("SELECT operation failed.")


def handle_project(args):
    """Handles the PROJECT command (π)."""
    usage = "Usage: project <input_table> <col1,col2,...> <output_table>"
    example = "Example: project students Name,Major projected_students"
    if len(args) != 3:
        print(usage); print(example); print("Error: Incorrect number of arguments.")
        if '#' in args: print("       (Ensure comments '#' are on separate lines or removed from the command)")
        return
    in_table, col_str, out_table = args

    if not out_table.isidentifier():
         print(f"Error: Invalid output table name '{out_table}'. Use letters, numbers, and underscores, not starting with a number.")
         return
    if out_table == in_table:
         print(f"Warning: Output table name '{out_table}' is the same as the input table. The input table will be overwritten.")

    if not _check_tables_exist([in_table], "PROJECT"): return

    columns_raw = _parse_columns(col_str)
    columns = [] # Unique columns
    seen_cols = set()
    for col in columns_raw:
        if col not in seen_cols: columns.append(col); seen_cols.add(col)

    if len(columns) != len(columns_raw): print(f"Warning: Duplicate columns removed from projection list. Using: {','.join(columns)}")
    if not columns or any(not col for col in columns): print("Error: Invalid or empty column list provided."); return

    print(f"Executing: {out_table} = π ({','.join(columns)}) ({in_table})")
    result = project(tables[in_table], columns) # Pass unique list
    if result is not None:
        tables[out_table] = result
        print(f"PROJECT operation complete. Result saved as '{out_table}' ({len(result)} rows).")
        if result.empty: print(result)
        else:
             rows_to_show = min(len(result), 10)
             print(f"First {rows_to_show} rows of '{out_table}':")
             print(result.head(rows_to_show).to_string())
    else:
        print("PROJECT operation failed.")


def handle_rename(args):
    """Handles the RENAME command."""
    usage = "Usage: rename <input_table> <old1:new1,old2:new2,...> <output_table>"
    example = "Example: rename students StudentID:SID,Name:FullName students_renamed"
    if len(args) != 3:
        print(usage); print(example); print("Error: Incorrect number of arguments.")
        if '#' in args: print("       (Ensure comments '#' are on separate lines or removed from the command)")
        return
    in_table, map_str, out_table = args

    if not out_table.isidentifier():
         print(f"Error: Invalid output table name '{out_table}'. Use letters, numbers, and underscores, not starting with a number.")
         return
    if out_table == in_table:
         print(f"Warning: Output table name '{out_table}' is the same as the input table. The input table will be overwritten.")

    if not _check_tables_exist([in_table], "RENAME"): return

    rename_map = _parse_rename_map(map_str)
    if rename_map is None: return # Error handled in parse function

    rename_str = ", ".join([f"{o}:{n}" for o, n in rename_map.items()])
    print(f"Executing: {out_table} = ρ ({rename_str}) ({in_table})")
    result = rename(tables[in_table], rename_map)
    if result is not None:
        tables[out_table] = result
        print(f"RENAME operation complete. Result saved as '{out_table}' ({len(result)} rows).")
        if result.empty: print(result)
        else:
             rows_to_show = min(len(result), 10)
             print(f"First {rows_to_show} rows of '{out_table}':")
             print(result.head(rows_to_show).to_string())
    else:
        print("RENAME operation failed.")


def handle_cartesian_product(args): _generic_binary_handler(args, "cp", cartesian_product, "×")
def handle_union(args): _generic_binary_handler(args, "union", union, "∪")
def handle_difference(args): _generic_binary_handler(args, "diff", difference, "-")
def handle_intersection(args): _generic_binary_handler(args, "intersect", intersection, "∩")
def handle_natural_join(args): _generic_binary_handler(args, "join", natural_join, "⨝")
def handle_division(args): _generic_binary_handler(args, "divide", division, "÷")

# --- Source Command Implementation ---

def execute_command_line(cmd_line):
    """ Helper function to parse and execute a single command line string from source file."""
    try:
        cmd_line = cmd_line.strip()
        if not cmd_line or cmd_line.startswith('#'): return True # Success for empty/comment

        parts = shlex.split(cmd_line)
        if not parts: return True

        command = parts[0].lower()
        args = parts[1:]

        if command in COMMANDS:
            # Check if the command is 'quit' or 'exit' - prevent from source file
            if command in ['quit', 'exit']:
                 print(f"Error (from source file): Cannot execute '{command}' command from within a source file.")
                 return False
            COMMANDS[command](args) # Execute
            return True # Assume success if no exception (handlers print errors)
        else:
            print(f"Error (from source file): Unknown command '{command}'.")
            return False
    except ValueError as e_shlex: print(f"Error parsing line: {e_shlex}\n  Line: '{cmd_line}'"); return False
    except Exception as e_exec: print(f"Error executing line: {e_exec}\n  Line: '{cmd_line}'"); traceback.print_exc(); return False


def handle_source(args):
    """Handles the SOURCE command to execute commands from a file."""
    usage = "Usage: source <filename>"
    if len(args) != 1: print(usage); print("Error: Incorrect number of arguments."); return

    filename = args[0]
    if not os.path.exists(filename): print(f"Error: Source file not found '{filename}'"); return

    print(f"--- Executing commands from '{filename}' ---")
    line_number, success_count, fail_count = 0, 0, 0
    stop_on_error = False # Set to True to abort sourcing on first error
    try:
        with open(filename, 'r') as f:
            for line in f:
                line_number += 1
                print(f"SRC {line_number}> {line.strip()}")
                if execute_command_line(line): success_count +=1
                else:
                    fail_count += 1
                    if stop_on_error:
                         print(f"--- Execution stopped due to error in '{filename}' at line {line_number} ---"); return
    except Exception as e: print(f"Error reading or processing source file '{filename}': {e}"); fail_count += 1

    print(f"--- Finished executing '{filename}' ---")
    print(f"    Commands attempted: {line_number}, Successful: {success_count}, Failed: {fail_count}")

# --- Help and Command Mapping ---

def handle_help(args=None):
    """Displays available commands."""
    print("""Available commands:
  load <file.csv> <table_name>   - Load data from CSV (creates empty if file missing)
  show                           - List loaded tables
  show <table_name>              - Display table content
  show schema <table_name>       - Display table schema
--- Basic Relational Algebra Operators ---
  select <in> "<cond>" <out>    - σ (Selection)
  project <in> <cols> <out>      - π (Projection)
  rename <in> <old:new,...> <out> - ρ (Rename)
  cp <in1> <in2> <out>           - × (Cartesian Product)
  union <in1> <in2> <out>        - ∪ (Set Union)
  diff <in1> <in2> <out>         - - (Set Difference, in1 - in2)
--- Additional Operators ---
  intersect <in1> <in2> <out>    - ∩ (Set Intersection)
  join <in1> <in2> <out>         - ⨝ (Natural Join)
  divide <dividend> <divisor> <out> - ÷ (Division, dividend / divisor)
--- Other ---
  source <filename>              - Execute commands from a file
  help                           - Show this help message
  quit / exit                    - Exit the program

Notes:
  - Table/column names are sanitized on load (spaces/_ -> _, etc.).
  - Use quotes for string literals in select conditions (e.g., "Major == 'CS'").
  - Comments (#) should be on separate lines in source files or CLI.""")

# Command mapping dictionary
COMMANDS = {
    "load": handle_load,
    "show": handle_show,
    "select": handle_select,    # σ
    "project": handle_project,  # π
    "rename": handle_rename,    # ρ
    "cp": handle_cartesian_product, # ×
    "union": handle_union,      # ∪
    "diff": handle_difference,  # - (Set Difference)
    "intersect": handle_intersection, # ∩
    "join": handle_natural_join,    # ⨝ (Natural Join)
    "divide": handle_division,      # ÷
    "source": handle_source,    # New command
    "help": handle_help,
    "quit": handle_quit,
    "exit": handle_quit,        # Alias for quit
}


def main():
    """Main loop for the CLI."""
    print("Relational Algebra CLI")
    print("Enter 'help' for commands, 'quit' or 'exit' to exit.")

    while True:
        try:
            cmd_line = input("RA> ")
            # Use the same execution helper as 'source' for consistency
            execute_command_line(cmd_line)

        except EOFError: print("\nQuitting..."); break # Handle Ctrl+D
        except KeyboardInterrupt: print("\nQuitting..."); break # Handle Ctrl+C
        except Exception as e: # Catch unexpected errors in the main loop itself
            print(f"\nAn unexpected error occurred in CLI loop: {e}")
            traceback.print_exc()

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Ensure Dummy CSVs Exist for Testing ---
    data_files = {
        "students.csv": pd.DataFrame({
            'StudentID': [101, 102, 103, 104, 105], 'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Major': ['CS', 'EE', 'CS', 'MATH', 'EE'], 'Age': [20, 21, 20, 22, 21] }),
        "enrollment.csv": pd.DataFrame({
             'StudentID': [101, 101, 102, 103, 103, 104, 105, 105],
             'CourseID': ['CS101', 'CS202', 'EE101', 'CS101', 'MATH201', 'MATH201','EE205','CS202'],
             'Grade': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'A'] }),
        "courses.csv": pd.DataFrame({
             'CourseID': ['CS101', 'CS202', 'EE101', 'MATH201', 'EE205'],
             'CourseName': ['Intro CS', 'Data Structures', 'Circuit Theory', 'Calculus III', 'Signals'],
             'Credits': [3, 4, 3, 4, 3] }),
        "core_cs.csv": pd.DataFrame({ 'CourseID': ['CS101', 'CS202'] }),
         "athletes.csv": pd.DataFrame({
            'StudentID': [102, 104, 106, 107], 'Name': ['Bob', 'David', 'Frank', 'Grace'],
            'Major': ['EE', 'MATH', 'CS', 'PHYS'], 'Age': [21, 22, 23, 20] })
    }
    for filename, df in data_files.items():
         if not os.path.exists(filename):
              try: df.to_csv(filename, index=False); print(f"Created dummy data file: {filename}")
              except Exception as e: print(f"Warning: Could not create dummy file {filename}: {e}")
    # --- End of Dummy Data Creation ---

    main() # Start the CLI