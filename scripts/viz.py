#%%
import optuna
from optuna import Study, load_study, visualization, importance

study_summaries = optuna.study.get_all_study_summaries(storage="postgresql://bestmann:deepquintic@server/deep_quintic")
for study in study_summaries:
    print(f"{study.study_name}\n  {study.n_trials}\n  {study.datetime_start}")

study = load_study('dynup-engine',  storage='postgresql://bestmann:deepquintic@server/deep_quintic')
df = study.trials_dataframe()
#print(df)
print(df.keys())
print(study.trials[0].params.keys())
len(df)

#%%

fig = visualization.plot_optimization_history(study)
fig.write_image("/tmp/study_fig.pdf")
exit(0)
#%%

df['state'].value_counts()
print("best set:")
df_sorted = df.sort_values(['value'], ascending=True)
#print(df_sorted)
best_set = df_sorted.iloc[0]
print("value " + str(best_set.value))
print("number " + str(best_set.number))
print('hyperparams:\n')
best_set = best_set.drop(['number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state'])
hyperparams = {}
policy_kwargs= {}
for param in best_set.keys():
    name = param[7:]
    hyperparams[name]=best_set[param]
    print(F"{name}: {best_set[param]}, ", end='\n')

#%%

#visualization.plot_contour(study, params=['freq', 'double_support_ratio'])
#visualization.plot_contour(study, params=['trunk_x_offset', 'trunk_pitch'])
#visualization.plot_contour(study, params=['foot_z_pause', 'trunk_pause'])
#visualization.plot_contour(study, params=['trunk_swing', 'trunk_pause'])

#%%

visualization.plot_slice(study).show()#, params=['freq', 'double_support_ratio', 'trunk_height'])

visualization.plot_contour(study, params=['double_support_ratio', 'trunk_swing']).show()

visualization.plot_optimization_history(study).show()

visualization.plot_parallel_coordinate(study).show()

#%%

df_sorted

#%%

print(importance.get_param_importances(study))


#visualization.plot_param_importances(study)
