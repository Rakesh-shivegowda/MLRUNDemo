from os import path
import mlrun

# Set the base project name
project_name_base = 'mlrun_demo'

# Initialize the MLRun project object
project = mlrun.get_or_create_project(project_name_base, context="./", user_project=True)

# Display the current project name
project_name = project.metadata.name
print(f'Full project name: {project_name}')

# mlrun: start-code

import mlrun
def prep_data(context, source_url: mlrun.DataItem, label_column='label'):

    # Convert the DataItem to a pandas DataFrame
    df = source_url.as_df()
    df[label_column] = df[label_column].astype('category').cat.codes    
    
    # Log the DataFrame size after the run
    context.log_result('num_rows', df.shape[0])

    # Store the dataset in your artifacts database
    context.log_dataset('cleaned_data', df=df, index=False, format='csv')
# mlrun: end-code

# Convert the local prep_data function to an MLRun project function
data_prep_func = mlrun.code_to_function(name='prep_data', kind='job', image='mlrun/mlrun')

# Set the source-data URL
source_url = mlrun.get_sample_path("data/iris/iris.data.raw.csv")

# Run the `data_prep_func` MLRun function locally
prep_data_run = data_prep_func.run(name='prep_data',
                                   handler=prep_data,
                                   inputs={'source_url': source_url},
                                   local=True)

print(prep_data_run.state())
dataset = prep_data_run.artifact('cleaned_data').as_df()
print(dataset.head())
