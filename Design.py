import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, io, base64, math

def df_info(df, descriptions=None, categorical_threshold=10):
    if descriptions is not None and len(descriptions)!=len(df.columns): raise ValueError(f"تعداد توضیحات ({len(descriptions)}) باید با تعداد ستون‌ها ({len(df.columns)}) برابر باشد.")
    data_types=['Categorical' if (pd.api.types.is_bool_dtype(s:=df[col]) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype,pd.CategoricalDtype) or (pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True)<categorical_threshold)) else 'Numerical' for col in df.columns]
    for col,t in zip(df.columns,data_types):
        if t=='Categorical': df[col]=df[col].astype('category')
    distinct=[', '.join(map(str,df[col].dropna().unique()[:30])) + (' ...' if df[col].nunique(dropna=True)>30 else '') if t=='Categorical' else ('Discrete' if df[col].nunique(dropna=True)<categorical_threshold else 'Continuous') for col,t in zip(df.columns,data_types)]
    result=pd.DataFrame({'ColumnName':df.columns,'Distribution':'','DataType':data_types,'count':df.notna().sum().values,'Distinct':distinct,'UniqueValues':df.nunique(dropna=True).values,'Missings':df.isna().sum().values,'MissingPercent':(df.isna().mean()*100).round(2).values,'mean':np.nan,'Mode':pd.Series([None]*len(df.columns),dtype='object'),'median':np.nan,'std':np.nan,'min':np.nan,'25%':np.nan,'50%':np.nan,'75%':np.nan,'max':np.nan})
    if descriptions is not None: result.insert(1,'Description',descriptions)
    for i,col in enumerate(df.columns):
        s=df[col]; modes=s.mode(dropna=True)
        result.loc[i,'Mode']=f'{len(modes)} modes: {", ".join(map(str,modes.iloc[:30].tolist()))}{" ..." if len(modes)>30 else ""}' if len(modes)>30 else (', '.join(map(str,modes.tolist())) if not modes.empty else np.nan)
        if data_types[i]=='Numerical': result.loc[i,['mean','median','std','min','25%','50%','75%','max']]=[s.mean(),s.median(),s.std(),s.min(),s.quantile(.25),s.median(),s.quantile(.75),s.max()]
        fig,ax=plt.subplots(figsize=(3.5,1.5))
        if data_types[i]=='Numerical': s.dropna().plot.hist(ax=ax,bins=15,edgecolor='white',linewidth=.5)
        else:
            counts=s.dropna().astype(str).value_counts().head(10); ax.bar(range(len(counts)),counts.values); ax.set_xticks(range(len(counts))); ax.set_xticklabels(counts.index,rotation=45,ha='right',fontsize=7)
        ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(axis='both',labelsize=7); ax.grid(axis='y',alpha=.2); plt.tight_layout()
        buffer=io.BytesIO(); fig.savefig(buffer,format='png',dpi=100,bbox_inches='tight'); plt.close(fig)
        result.loc[i,'Distribution']=f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}" style="width:220px;height:90px;">'
    pd.set_option('display.max_columns',None); pd.set_option('display.max_colwidth',None); pd.set_option('display.width',None)
    styler=result.style.format({'Distribution':lambda x:x,'MissingPercent':'{:.2f} %','mean':'{:.2f}','median':'{:.2f}','std':'{:.2f}','min':'{:.2f}','25%':'{:.2f}','50%':'{:.2f}','75%':'{:.2f}','max':'{:.2f}'},escape=None).set_properties(**{'text-align':'center','vertical-align':'middle'}).set_table_styles([{'selector':'th','props':[('text-align','center'),('font-weight','bold')]}]).hide(axis='index')
    if descriptions is not None: styler=styler.set_properties(subset=['Description'],**{'text-align':'right','direction':'rtl','white-space':'pre-wrap','max-width':'400px'})
    return styler


def numerical_plot(df, numericals):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    n = len(numericals)
    if n == 0: return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(18, 6 * nrows))
    outer = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35, wspace=0.25)
    MEAN_COLOR = "#DC2626"
    MEDIAN_COLOR = "#2563EB"
    MODE_COLOR = "#16A34A"
    IQR_COLOR = "#6B7280"
    legend_elements = [Line2D([0], [0], color=MEAN_COLOR, lw=1.5, label=r"$\mu$"), Line2D([0], [0], color=MEDIAN_COLOR, lw=1.5, label=r"$Me$"), Line2D([0], [0], color=MODE_COLOR, lw=1.5, label=r"$Mo$"), Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=IQR_COLOR, markersize=5, label="IQR Outlier")]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.99))
    plt.subplots_adjust(top=0.93)
    for i, feature in enumerate(numericals):
        row, col = i // ncols, i % ncols
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        ax_hist = fig.add_subplot(inner[0])
        ax_box = fig.add_subplot(inner[1], sharex=ax_hist)
        s = df[feature].dropna()
        if s.empty: continue
        mean_val = s.mean()
        median_val = s.median()
        mode_val = s.mode().iloc[0]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        L = q1 - 1.5 * iqr
        U = q3 + 1.5 * iqr
        iqr_outliers = s[(s < L) | (s > U)]
        sns.histplot(s, bins="auto", kde=True, ax=ax_hist)
        ax_hist.set_title(feature, fontsize=13, fontweight="bold")
        ax_hist.set_xlabel(feature)
        ax_hist.set_ylabel("")
        ax_hist.grid(False)
        ax_hist.axvline(mean_val, color=MEAN_COLOR, linewidth=1.2)
        ax_hist.axvline(median_val, color=MEDIAN_COLOR, linewidth=1.2)
        ax_hist.axvline(mode_val, color=MODE_COLOR, linewidth=1.2)
        sns.boxplot(x=s, ax=ax_box, orient="h", width=0.5, color="#93C5FD", linewidth=1.3, showfliers=False)
        if len(iqr_outliers) > 0: ax_box.scatter(iqr_outliers, np.zeros(len(iqr_outliers)), s=18, facecolors="none", edgecolors=IQR_COLOR, linewidths=0.8, zorder=5)
        ax_box.axvline(mean_val, color=MEAN_COLOR, linewidth=1.5, zorder=6)
        ax_box.axvline(median_val, color=MEDIAN_COLOR, linewidth=1.5, zorder=6)
        ax_box.axvline(mode_val, color=MODE_COLOR, linewidth=1.5, zorder=6)
        ax_box.set_xlabel("")
        ax_box.set_ylabel("")
        ax_box.set_yticks([])
        ax_box.tick_params(axis="x", labelbottom=False)
        for spine in ax_box.spines.values(): spine.set_visible(False)
    plt.show()
    plt.close(fig)
    
    
def categorical_plot(df, categoricals):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(categoricals)
    if n == 0:
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(18, 5 * nrows)
    )

    axes = np.array(axes).flatten()

    for ax, feature in zip(axes, categoricals):

        s = df[feature].dropna()
        counts = s.value_counts()
        percentages = counts / counts.sum() * 100

        sns.barplot(
            x=counts.index,
            y=counts.values,
            ax=ax
        )

        for i, (count, percent) in enumerate(zip(counts, percentages)):
            ax.text(
                i, count,
                f'{count:,}\n({percent:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_title(feature, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(False)

    for ax in axes[n:]:
        ax.remove()

    plt.tight_layout()
    plt.show()
    plt.close(fig)