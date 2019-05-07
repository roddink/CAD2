import xgboost as xgb
import matplotlib 
matplotlib.use("agg")
bts = xgb.Booster()
bts.load_model("0001.model")
ax = xgb.plot_tree(bts,num_trees=1,rankdir = "LR")
fig = ax.get_figure()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')
ax2 = xgb.plot_importance(bts)
fig2 = ax2.get_figure()
fig2.set_size_inches(150, 100)
fig2.savefig('tree_importance.png')

