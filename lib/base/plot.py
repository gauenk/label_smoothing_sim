def add_legend(ax,legend_handles,legend_str,legend_title):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(legend_handles,legend_str,
              title = legend_title,
              title_fontsize=15,
              fontsize=15,
              loc='center left',
              bbox_to_anchor=(1, 0.5))    
    return ax
