"""Exploratory Data Analysis for DOHA Security Clearance Decisions

Focus: Outcome distribution, guideline frequency, text length analysis
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

dataDir = Path(__file__).resolve().parents[1] / "Challenge_Data"

#Helper to print stats
def printStats(data, label):
    print(f"\n{label}:")
    print(f"  Min:    {data.min():,}")
    print(f"  Max:    {data.max():,}")
    print(f"  Mean:   {data.mean():,.0f}")
    print(f"  Median: {data.median():,.0f}")
    print(f"  StdDev: {data.std():,.0f}")


def loadData():
    files = sorted(glob.glob(str(dataDir / "cases_clean_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {dataDir}")
    
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"\nLoaded {df.shape[0]:,} rows, {df.shape[1]} cols\n")
    return df


def analyzeOutcomes(df):
    print("OUTCOME DISTRIBUTION")
    print("-" * 50)
    
    total = len(df)
    counts = df['outcome'].value_counts()
    
    print("\nFrequencies:")
    for outcome, count in counts.items():
        pct = count / total * 100
        print(f"  {outcome:15s}: {count:6,} ({pct:5.2f}%)")
    
    missing = df['outcome'].isna().sum()
    if missing > 0:
        print(f"  {'Missing':15s}: {missing:6,} ({missing/total*100:.2f}%)")
    
    print("\nBy case type:")
    crosstab = pd.crosstab(df['case_type'], df['outcome'], normalize='index') * 100
    print(crosstab.round(2).to_string())
    
    #Charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Outcome Distribution')
    axes[0].set_ylabel('Count')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    for i, v in enumerate(counts):
        axes[0].text(i, v + 500, f'{v:,}', ha='center')
    
    #Pie with legend - only show % for slices > 1%
    def showPercent(pct):
        return f'{pct:.1f}%' if pct > 1 else ''
    
    colors = sns.color_palette("Set2", len(counts))
    wedges, texts, autotexts = axes[1].pie(counts, autopct=showPercent, 
                                             startangle=140, colors=colors,
                                             textprops={'fontsize': 11})
    axes[1].legend(wedges, [f'{k}: {v:,} ({v/total*100:.1f}%)' for k, v in counts.items()],
                   loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=10)
    axes[1].set_title('Outcome Proportions')
    
    plt.tight_layout()
    plt.savefig(dataDir.parent / 'notebooks' / 'outcome_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved: outcome_distribution.png")
    plt.close()
    
    return counts


def analyzeGuidelines(df):
    print(f"\n{'='*50}")
    print("GUIDELINE FREQUENCY")
    print("-" * 50)
    
    #Only use formal rulings (high quality labels)
    formal = df[df['label_source'] == 'formal_ruling']
    other = df[df['label_source'] != 'formal_ruling']
    
    print(f"\nLabel breakdown:")
    print(f"  Formal ruling: {len(formal):,}")
    print(f"  Other sources: {len(other):,}")
    
    guidelines = formal['guidelines'].explode().dropna()
    totalGuides = len(guidelines)
    casesWithGuides = formal['guidelines'].apply(lambda x: len(x) > 0 if hasattr(x, '__len__') else False).sum()
    
    print(f"\nFormal cases with guidelines: {casesWithGuides:,}")
    print(f"Total instances: {totalGuides:,}")
    if casesWithGuides > 0:
        print(f"Avg per case: {totalGuides / casesWithGuides:.2f}")
    
    counts = guidelines.value_counts().sort_index()
    pcts = (counts / casesWithGuides * 100).round(2)
    
    names = {
        'A': 'Allegiance', 'B': 'Foreign Influence', 'C': 'Foreign Preference',
        'D': 'Sexual Behavior', 'E': 'Personal Conduct', 'F': 'Financial',
        'G': 'Alcohol', 'H': 'Drug Involvement', 'I': 'Psychological',
        'J': 'Criminal Conduct', 'K': 'Protected Info', 'L': 'Outside Activities',
        'M': 'Info Technology'
    }
    
    print("\nFrequencies (formal only):")
    for code in sorted(names.keys()):
        if code in counts.index:
            print(f"  {code} - {names[code]:20s}: {counts[code]:6,} ({pcts[code]:5.2f}%)")
    
    perCase = formal['guidelines'].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
    print(f"\nPer case stats:")
    print(f"  Min: {perCase.min()}, Max: {perCase.max()}")
    print(f"  Mean: {perCase.mean():.2f}, Median: {perCase.median():.0f}")
    
    print("\nLabel sources (all):")
    labelSrc = df['label_source'].value_counts()
    for src, cnt in labelSrc.items():
        print(f"  {src:25s}: {cnt:6,} ({cnt/len(df)*100:.2f}%)")
    
    #Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    counts.plot(kind='bar', ax=axes[0, 0], color='coral', edgecolor='black')
    axes[0, 0].set_title('Guideline Frequency - Formal Rulings')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(perCase, bins=range(0, perCase.max() + 2), 
                    color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Guidelines per Case')
    axes[0, 1].set_xlabel('Number of Guidelines')
    axes[0, 1].axvline(perCase.mean(), color='red', linestyle='--', 
                       label=f'Mean: {perCase.mean():.2f}')
    axes[0, 1].legend()
    
    top10 = counts.nlargest(10)
    axes[1, 0].barh(range(len(top10)), top10.values, color='teal', edgecolor='black')
    axes[1, 0].set_yticks(range(len(top10)))
    axes[1, 0].set_yticklabels([f"{g} - {names.get(g, 'Unknown')}" for g in top10.index])
    axes[1, 0].set_title('Top 10 Guidelines')
    axes[1, 0].invert_yaxis()
    
    #Fix pie chart with legend to avoid overlap
    def showPct(pct):
        return f'{pct:.1f}%' if pct > 5 else ''
    
    labelColors = {'formal_ruling': '#2ecc71', 'inherited_from_hearing': '#3498db', 
                   'regex_fallback': '#e74c3c', 'none': '#95a5a6'}
    colors = [labelColors.get(x, '#7f8c8d') for x in labelSrc.index]
    wedges, texts, autotexts = axes[1, 1].pie(labelSrc.values, autopct=showPct,
                                               startangle=90, colors=colors,
                                               textprops={'fontsize': 10})
    
    #Create cleaner legend with counts and percentages
    totalCases = labelSrc.sum()
    legendLabels = [f'{src}: {cnt:,} ({cnt/totalCases*100:.1f}%)' 
                    for src, cnt in labelSrc.items()]
    axes[1, 1].legend(wedges, legendLabels, loc="center left", 
                      bbox_to_anchor=(1, 0.5), fontsize=9)
    axes[1, 1].set_title('Label Quality Tiers')
    
    plt.tight_layout()
    plt.savefig(dataDir.parent / 'notebooks' / 'guideline_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: guideline_analysis.png")
    plt.close()
    
    return counts, labelSrc

def analyzeTextLengths(df):
    print(f"\n{'='*50}")
    print("TEXT LENGTH ANALYSIS")
    print("-" * 50)
    
    #Fix: exclude empty strings
    hasText = df['text_features'].str.strip().str.len() > 0
    emptyCount = (~hasText).sum()
    
    print(f"\nQuality check:")
    print(f"  Empty: {emptyCount:,} ({emptyCount/len(df)*100:.2f}%)")
    print(f"  Valid: {hasText.sum():,} ({hasText.sum()/len(df)*100:.2f}%)")
    
    fullLen = df['full_text'].str.len()
    printStats(fullLen, "Full text (chars)")
    
    print("\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th: {fullLen.quantile(p/100):,.0f}")
    
    #Only valid cases
    valid = df[hasText]
    featLen = valid['text_features'].str.len()
    
    print(f"\nText features (valid only, {len(featLen):,} cases):")
    printStats(featLen, "Features")
    
    fullWords = df['full_text'].str.split().str.len()
    featWords = valid['text_features'].str.split().str.len()
    
    print(f"\nWord counts:")
    print(f"  Full text:  {fullWords.mean():,.0f}")
    print(f"  Features:   {featWords.mean():,.0f}")
    print(f"  Reduction:  {(1 - featWords.mean()/fullWords.mean())*100:.1f}%")
    
    print("\nLength by outcome:")
    byOutcome = df.groupby('outcome')['full_text'].apply(lambda x: x.str.len().mean())
    for outcome, length in byOutcome.items():
        print(f"  {outcome:12s}: {length:,.0f} chars")
    
    #Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(fullLen / 1000, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Full Text Length')
    axes[0, 0].set_xlabel('Length (thousands of chars)')
    axes[0, 0].axvline(fullLen.mean() / 1000, color='red', linestyle='--', 
                       label=f'Mean: {fullLen.mean()/1000:.1f}k')
    axes[0, 0].legend()
    
    axes[0, 1].hist(featLen / 1000, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Features Length (Valid Only)')
    axes[0, 1].set_xlabel('Length (thousands of chars)')
    axes[0, 1].axvline(featLen.mean() / 1000, color='red', linestyle='--', 
                       label=f'Mean: {featLen.mean()/1000:.1f}k')
    axes[0, 1].legend()
    
    #Box by outcome
    outcomeData = []
    outcomeLabels = []
    for outcome in df['outcome'].dropna().unique():
        lengths = df[df['outcome'] == outcome]['full_text'].str.len() / 1000
        outcomeData.append(lengths)
        outcomeLabels.append(outcome)
    
    axes[1, 0].boxplot(outcomeData, labels=outcomeLabels, patch_artist=True)
    axes[1, 0].set_title('Length by Outcome')
    axes[1, 0].set_ylabel('Length (thousands of chars)')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    validFull = df.loc[hasText, 'full_text'].str.split().str.len()
    comparison = pd.DataFrame({'Full Text': validFull, 'Features': featWords})
    comparison.plot(kind='hist', bins=50, alpha=0.6, ax=axes[1, 1], edgecolor='black')
    axes[1, 1].set_title('Word Count Comparison')
    axes[1, 1].set_xlabel('Word Count')
    
    plt.tight_layout()
    plt.savefig(dataDir.parent / 'notebooks' / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: text_length_analysis.png")
    plt.close()
    
    return fullLen, featLen, emptyCount


def analyzeDataQuality(df):
    print(f"\n{'='*50}")
    print("DATA QUALITY")
    print("-" * 50)
    
    useTrain = df['use_for_training'].sum()
    formal = (df['label_source'] == 'formal_ruling').sum()
    ready = (df['use_for_training'] & (df['label_source'] == 'formal_ruling')).sum()
    
    print(f"\nML readiness:")
    print(f"  Training flag: {useTrain:,} ({useTrain/len(df)*100:.1f}%)")
    print(f"  Formal labels: {formal:,} ({formal/len(df)*100:.1f}%)")
    print(f"  Ready (both):  {ready:,} ({ready/len(df)*100:.1f}%)")
    
    print(f"\nField quality:")
    nullOutcome = df['outcome'].isna().sum()
    print(f"  outcome null:      {nullOutcome:6,} ({nullOutcome/len(df)*100:.2f}%)")
    
    emptyFeat = (df['text_features'].str.strip().str.len() == 0).sum()
    print(f"  text_features empty: {emptyFeat:6,} ({emptyFeat/len(df)*100:.2f}%)")
    
    nullGuide = df['guidelines'].isna().sum()
    print(f"  guidelines null:     {nullGuide:6,} ({nullGuide/len(df)*100:.2f}%)")
    
    splits = df['split_group'].nunique()
    print(f"\nSplitting:")
    print(f"  Split groups: {splits:,}")
    print(f"  Avg per group: {len(df)/splits:.1f}")
    
    print(f"\nRegime breakdown:")
    regimes = df[df['use_for_training']]['guideline_regime'].value_counts()
    for regime, cnt in regimes.items():
        print(f"  {regime:12s}: {cnt:6,} ({cnt/useTrain*100:.2f}%)")
    
    return ready


def generateSummary(df, outcomes, guidelines, fullLen, ready, emptyCount):
    print(f"\n{'='*50}")
    print("SUMMARY")
    print("=" * 50)
    
    print(f"\nOverview:")
    print(f"  Total cases:   {len(df):,}")
    print(f"  Training-ready: {ready:,}")
    print(f"  Date range:    {df['date'].min()} to {df['date'].max()}")
    
    print(f"\nOutcomes:")
    top = outcomes.index[0]
    print(f"  Most common:   {top} ({outcomes[top]:,})")
    print(f"  Imbalance:     {outcomes.max()/outcomes.min():.1f}:1")
    
    print(f"\nGuidelines (formal):")
    print(f"  Most frequent: {guidelines.index[0]} ({guidelines.iloc[0]:,})")
    print(f"  Least frequent: {guidelines.index[-1]} ({guidelines.iloc[-1]:,})")
    print(f"  Total types:   {len(guidelines)}")
    
    print(f"\nText:")
    print(f"  Avg length:    {fullLen.mean():,.0f} chars")
    print(f"  Avg words:     {df['full_text'].str.split().str.len().mean():,.0f}")
    print(f"  Empty features: {emptyCount:,}")
    
    print(f"\nFixes applied:")
    print(f"  - Empty strings excluded")
    print(f"  - Label tiers separated")
    print(f"  - Formal rulings isolated")
    
    print(f"\nFor ML pipeline:")
    print(f"  - Use text_features (leak-free)")
    print(f"  - Filter: use_for_training=True AND label_source='formal_ruling'")
    print(f"  - Exclude empty text_features")
    print(f"  - Group splits by split_group")
    print(f"  - Handle imbalance with stratification")
    
    print(f"\n{'='*50}\n")


def main():
    try:
        df = loadData()
        
        outcomes = analyzeOutcomes(df)
        guidelines, labels = analyzeGuidelines(df)
        fullLen, featLen, emptyCount = analyzeTextLengths(df)
        ready = analyzeDataQuality(df)
        
        generateSummary(df, outcomes, guidelines, fullLen, ready, emptyCount)
        
        print("EDA complete!")
        print(f"Charts saved in: {dataDir.parent / 'notebooks'}")
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
