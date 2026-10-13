"""Exploratory Data Analysis for DOHA Security Clearance Decisions

Focus: Outcome distribution, guideline frequency, text length analysis
Security: Uses cleaned Challenge_Data parquet files with proper data validation
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


def loadData():
    #Load all cleaned parquet shards
    parquetFiles = sorted(glob.glob(str(dataDir / "cases_clean_*.parquet")))
    if not parquetFiles:
        raise FileNotFoundError(f"No parquet files in {dataDir}")
    
    dataFrames = [pd.read_parquet(f) for f in parquetFiles]
    combinedData = pd.concat(dataFrames, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset loaded: {combinedData.shape[0]:,} rows × {combinedData.shape[1]} columns")
    print(f"{'='*60}\n")
    
    return combinedData


def analyzeOutcomes(dataFrame):
    #Outcome distribution analysis
    print("OUTCOME DISTRIBUTION ANALYSIS")
    print("-" * 60)
    
    outcomeData = dataFrame['outcome'].copy()
    totalCases = len(outcomeData)
    
    #Count valid outcomes only
    validOutcomes = outcomeData.dropna()
    outcomeCounts = validOutcomes.value_counts()
    outcomePercents = (outcomeCounts / totalCases * 100).round(2)
    
    print("\nOutcome frequencies:")
    for outcome, count in outcomeCounts.items():
        percent = outcomePercents[outcome]
        print(f"  {outcome:15s}: {count:6,} ({percent:5.2f}%)")
    
    #Missing values
    missingCount = outcomeData.isna().sum()
    if missingCount > 0:
        missingPercent = (missingCount / totalCases * 100).round(2)
        print(f"  {'Missing':15s}: {missingCount:6,} ({missingPercent:5.2f}%)")
    
    #Outcome by case type
    print("\nOutcome by case type:")
    caseTypeOutcome = pd.crosstab(
        dataFrame['case_type'], 
        dataFrame['outcome'], 
        normalize='index'
    ) * 100
    print(caseTypeOutcome.round(2).to_string())
    
    #Visualize outcomes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    #Bar chart
    outcomeCounts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Outcome Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Outcome', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(outcomeCounts):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)
    
    #Pie chart
    axes[1].pie(
        outcomeCounts, 
        labels=outcomeCounts.index, 
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("Set2")
    )
    axes[1].set_title('Outcome Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(dataDir.parent / 'notebooks' / 'outcome_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: outcome_distribution.png")
    plt.close()
    
    return outcomeCounts


def analyzeGuidelines(dataFrame):
    #Guideline frequency analysis
    print(f"\n{'='*60}")
    print("GUIDELINE FREQUENCY ANALYSIS")
    print("-" * 60)
    
    #Separate by label quality
    formalRulingData = dataFrame[dataFrame['label_source'] == 'formal_ruling']
    otherLabels = dataFrame[dataFrame['label_source'] != 'formal_ruling']
    
    print(f"\nLabel source breakdown:")
    print(f"  Formal ruling (authoritative): {len(formalRulingData):,}")
    print(f"  Other sources (lower quality):  {len(otherLabels):,}")
    
    #Analyze formal ruling guidelines only
    guidelineData = formalRulingData['guidelines'].explode()
    guidelineData = guidelineData[guidelineData.notna()]
    totalGuidelines = len(guidelineData)
    casesWithGuidelines = formalRulingData['guidelines'].apply(
        lambda x: len(x) > 0 if hasattr(x, '__len__') else False
    ).sum()
    
    print(f"\nFormal ruling cases with guidelines: {casesWithGuidelines:,}")
    print(f"Total guideline instances: {totalGuidelines:,}")
    if casesWithGuidelines > 0:
        print(f"Average guidelines per case: {totalGuidelines / casesWithGuidelines:.2f}")
    
    #Count each guideline
    guidelineCounts = guidelineData.value_counts().sort_index()
    guidelinePercents = (guidelineCounts / casesWithGuidelines * 100).round(2)
    
    print("\nGuideline frequencies (formal rulings only):")
    guidelineNames = {
        'A': 'Allegiance to US',
        'B': 'Foreign Influence',
        'C': 'Foreign Preference',
        'D': 'Sexual Behavior',
        'E': 'Personal Conduct',
        'F': 'Financial Considerations',
        'G': 'Alcohol Consumption',
        'H': 'Drug Involvement',
        'I': 'Psychological Conditions',
        'J': 'Criminal Conduct',
        'K': 'Handling Protected Info',
        'L': 'Outside Activities',
        'M': 'Info Technology Use'
    }
    
    for guideline in sorted(guidelineNames.keys()):
        if guideline in guidelineCounts.index:
            count = guidelineCounts[guideline]
            percent = guidelinePercents[guideline]
            name = guidelineNames[guideline]
            print(f"  {guideline} - {name:25s}: {count:6,} ({percent:5.2f}%)")
    
    #Guidelines per case distribution
    guidelinesPerCase = formalRulingData['guidelines'].apply(
        lambda x: len(x) if hasattr(x, '__len__') else 0
    )
    
    print(f"\nGuidelines per case statistics:")
    print(f"  Min:    {guidelinesPerCase.min()}")
    print(f"  Max:    {guidelinesPerCase.max()}")
    print(f"  Mean:   {guidelinesPerCase.mean():.2f}")
    print(f"  Median: {guidelinesPerCase.median():.0f}")
    
    #Label source quality distribution
    print("\nLabel source distribution (all data):")
    labelSource = dataFrame['label_source'].value_counts()
    for source, count in labelSource.items():
        percent = round(count / len(dataFrame) * 100, 2)
        print(f"  {source:25s}: {count:6,} ({percent:5.2f}%)")
    
    #Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    #Guideline frequency bar chart (formal only)
    guidelineCounts.plot(kind='bar', ax=axes[0, 0], color='coral', edgecolor='black')
    axes[0, 0].set_title('Guideline Frequency - Formal Rulings Only', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Guideline', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    #Guidelines per case histogram
    axes[0, 1].hist(guidelinesPerCase, bins=range(0, guidelinesPerCase.max() + 2), 
                    color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Guidelines per Case - Formal Rulings', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Guidelines', fontsize=12)
    axes[0, 1].set_ylabel('Case Count', fontsize=12)
    axes[0, 1].axvline(guidelinesPerCase.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {guidelinesPerCase.mean():.2f}')
    axes[0, 1].legend()
    
    #Top 10 guidelines horizontal bar
    topGuidelines = guidelineCounts.nlargest(10)
    axes[1, 0].barh(range(len(topGuidelines)), topGuidelines.values, color='teal', edgecolor='black')
    axes[1, 0].set_yticks(range(len(topGuidelines)))
    axes[1, 0].set_yticklabels([f"{g} - {guidelineNames.get(g, 'Unknown')}" for g in topGuidelines.index])
    axes[1, 0].set_title('Top 10 Guidelines - Formal Rulings', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Count', fontsize=12)
    axes[1, 0].invert_yaxis()
    
    #Label source quality breakdown
    labelColors = {'formal_ruling': '#2ecc71', 'inherited_from_hearing': '#3498db', 
                   'regex_fallback': '#e74c3c', 'none': '#95a5a6'}
    colors = [labelColors.get(x, '#7f8c8d') for x in labelSource.index]
    axes[1, 1].pie(
        labelSource.values, 
        labels=[f"{x}\n({v:,})" for x, v in zip(labelSource.index, labelSource.values)], 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    axes[1, 1].set_title('Label Source Quality Tiers', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(dataDir.parent / 'notebooks' / 'guideline_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: guideline_analysis.png")
    plt.close()
    
    return guidelineCounts, labelSource


def analyzeTextLengths(dataFrame):
    #Text length analysis for ML features
    print(f"\n{'='*60}")
    print("TEXT LENGTH ANALYSIS")
    print("-" * 60)
    
    #Filter out empty strings properly
    validTextFeatures = dataFrame['text_features'].str.strip().str.len() > 0
    emptyCount = (~validTextFeatures).sum()
    
    print(f"\nData quality check:")
    print(f"  Empty text_features: {emptyCount:,} ({emptyCount / len(dataFrame) * 100:.2f}%)")
    print(f"  Valid text_features: {validTextFeatures.sum():,} ({validTextFeatures.sum() / len(dataFrame) * 100:.2f}%)")
    
    #Full text length
    fullTextLengths = dataFrame['full_text'].str.len()
    
    print("\nFull text statistics (characters):")
    print(f"  Min:     {fullTextLengths.min():,}")
    print(f"  Max:     {fullTextLengths.max():,}")
    print(f"  Mean:    {fullTextLengths.mean():,.0f}")
    print(f"  Median:  {fullTextLengths.median():,.0f}")
    print(f"  Std Dev: {fullTextLengths.std():,.0f}")
    
    #Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        value = fullTextLengths.quantile(p / 100)
        print(f"  {p:2d}th: {value:,.0f}")
    
    #Text features length (leak-free, excluding empty)
    validData = dataFrame[validTextFeatures]
    textFeaturesLengths = validData['text_features'].str.len()
    
    print(f"\nText features statistics (valid only, {len(textFeaturesLengths):,} cases):")
    print(f"  Min:     {textFeaturesLengths.min():,}")
    print(f"  Max:     {textFeaturesLengths.max():,}")
    print(f"  Mean:    {textFeaturesLengths.mean():,.0f}")
    print(f"  Median:  {textFeaturesLengths.median():,.0f}")
    print(f"  Std Dev: {textFeaturesLengths.std():,.0f}")
    
    #Word counts
    fullTextWords = dataFrame['full_text'].str.split().str.len()
    textFeaturesWords = validData['text_features'].str.split().str.len()
    
    print(f"\nWord counts:")
    print(f"  Full text mean:     {fullTextWords.mean():,.0f}")
    print(f"  Text features mean: {textFeaturesWords.mean():,.0f}")
    print(f"  Reduction:          {(1 - textFeaturesWords.mean() / fullTextWords.mean()) * 100:.1f}%")
    
    #Text length by outcome
    print("\nMean text length by outcome:")
    lengthByOutcome = dataFrame.groupby('outcome')['full_text'].apply(lambda x: x.str.len().mean())
    for outcome, length in lengthByOutcome.items():
        print(f"  {outcome:15s}: {length:,.0f} chars")
    
    #Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    #Full text length histogram
    axes[0, 0].hist(fullTextLengths / 1000, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Full Text Length Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Length (thousands of characters)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].axvline(fullTextLengths.mean() / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {fullTextLengths.mean() / 1000:.1f}k')
    axes[0, 0].legend()
    
    #Text features length histogram (valid only)
    axes[0, 1].hist(textFeaturesLengths / 1000, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Text Features Length (Valid Only)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Length (thousands of characters)', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].axvline(textFeaturesLengths.mean() / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {textFeaturesLengths.mean() / 1000:.1f}k')
    axes[0, 1].legend()
    
    #Box plot by outcome
    outcomeData = []
    outcomeLabels = []
    for outcome in dataFrame['outcome'].dropna().unique():
        lengths = dataFrame[dataFrame['outcome'] == outcome]['full_text'].str.len() / 1000
        outcomeData.append(lengths)
        outcomeLabels.append(outcome)
    
    axes[1, 0].boxplot(outcomeData, labels=outcomeLabels, patch_artist=True)
    axes[1, 0].set_title('Text Length by Outcome', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Outcome', fontsize=12)
    axes[1, 0].set_ylabel('Length (thousands of characters)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    #Word count comparison (valid data only)
    validFullText = dataFrame.loc[validTextFeatures, 'full_text'].str.split().str.len()
    comparisonData = pd.DataFrame({
        'Full Text': validFullText,
        'Features Only': textFeaturesWords
    })
    comparisonData.plot(kind='hist', bins=50, alpha=0.6, ax=axes[1, 1], edgecolor='black')
    axes[1, 1].set_title('Word Count Comparison (Valid Data)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Word Count', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].legend(['Full Text', 'Features Only'])
    
    plt.tight_layout()
    plt.savefig(dataDir.parent / 'notebooks' / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: text_length_analysis.png")
    plt.close()
    
    return fullTextLengths, textFeaturesLengths, emptyCount


def analyzeDataQuality(dataFrame):
    #Data quality checks for ML readiness
    print(f"\n{'='*60}")
    print("DATA QUALITY ANALYSIS")
    print("-" * 60)
    
    #Training data availability
    useForTraining = dataFrame['use_for_training'].sum()
    formalRuling = (dataFrame['label_source'] == 'formal_ruling').sum()
    trainingReady = (dataFrame['use_for_training'] & (dataFrame['label_source'] == 'formal_ruling')).sum()
    
    print(f"\nML training readiness:")
    print(f"  Use for training flag:     {useForTraining:,} ({useForTraining / len(dataFrame) * 100:.1f}%)")
    print(f"  Formal ruling labels:      {formalRuling:,} ({formalRuling / len(dataFrame) * 100:.1f}%)")
    print(f"  Training-ready (both):     {trainingReady:,} ({trainingReady / len(dataFrame) * 100:.1f}%)")
    
    #Missing and empty critical fields
    print(f"\nCritical field quality:")
    
    outcomeNull = dataFrame['outcome'].isna().sum()
    print(f"  outcome (null):         {outcomeNull:6,} ({outcomeNull / len(dataFrame) * 100:.2f}%)")
    
    textFeaturesEmpty = (dataFrame['text_features'].str.strip().str.len() == 0).sum()
    print(f"  text_features (empty):  {textFeaturesEmpty:6,} ({textFeaturesEmpty / len(dataFrame) * 100:.2f}%)")
    
    guidelinesNull = dataFrame['guidelines'].isna().sum()
    print(f"  guidelines (null):      {guidelinesNull:6,} ({guidelinesNull / len(dataFrame) * 100:.2f}%)")
    
    regimeNull = dataFrame['guideline_regime'].isna().sum()
    print(f"  guideline_regime (null):{regimeNull:6,} ({regimeNull / len(dataFrame) * 100:.2f}%)")
    
    #Split groups
    splitGroups = dataFrame['split_group'].nunique()
    print(f"\nData splitting:")
    print(f"  Unique split groups: {splitGroups:,}")
    print(f"  Cases per group:     {len(dataFrame) / splitGroups:.1f} avg")
    
    #Guideline regime balance
    print(f"\nGuideline regime for training:")
    regimeTraining = dataFrame[dataFrame['use_for_training']]['guideline_regime'].value_counts()
    for regime, count in regimeTraining.items():
        percent = round(count / useForTraining * 100, 2)
        print(f"  {regime:15s}: {count:6,} ({percent:5.2f}%)")
    
    return trainingReady


def generateSummary(dataFrame, outcomeCounts, guidelineCounts, fullTextLengths, trainingReady, emptyCount):
    #Generate comprehensive summary report
    print(f"\n{'='*60}")
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset Overview:")
    print(f"  Total cases:            {len(dataFrame):,}")
    print(f"  Training-ready cases:   {trainingReady:,}")
    print(f"  Date range:             {dataFrame['date'].min()} to {dataFrame['date'].max()}")
    
    print(f"\nOutcome Summary:")
    topOutcome = outcomeCounts.index[0]
    print(f"  Most common:            {topOutcome} ({outcomeCounts[topOutcome]:,} cases)")
    print(f"  Class imbalance ratio:  {outcomeCounts.max() / outcomeCounts.min():.2f}:1")
    
    print(f"\nGuideline Summary (Formal Rulings):")
    print(f"  Most frequent:          {guidelineCounts.index[0]} ({guidelineCounts.iloc[0]:,} cases)")
    print(f"  Least frequent:         {guidelineCounts.index[-1]} ({guidelineCounts.iloc[-1]:,} cases)")
    print(f"  Total guideline types:  {len(guidelineCounts)}")
    
    print(f"\nText Summary:")
    print(f"  Avg document length:    {fullTextLengths.mean():,.0f} characters")
    print(f"  Avg word count:         {dataFrame['full_text'].str.split().str.len().mean():,.0f} words")
    print(f"  Empty text_features:    {emptyCount:,} cases")
    
    print(f"\nCritical Issues Fixed:")
    print(f"  ✓ Empty strings now excluded from text analysis")
    print(f"  ✓ Label tiers separated in visualizations")
    print(f"  ✓ Formal rulings isolated for guideline analysis")
    
    print(f"\nML Pipeline Recommendations:")
    print(f"  ✓ Use text_features column (leak-free)")
    print(f"  ✓ Filter: use_for_training=True AND label_source='formal_ruling'")
    print(f"  ✓ Exclude cases with empty text_features")
    print(f"  ✓ Group splits by split_group column")
    print(f"  ✓ Handle class imbalance with stratification")
    
    print(f"\n{'='*60}\n")


def main():
    #Execute complete EDA pipeline
    try:
        #Load data
        dataFrame = loadData()
        
        #Run analyses
        outcomeCounts = analyzeOutcomes(dataFrame)
        guidelineCounts, labelSource = analyzeGuidelines(dataFrame)
        fullTextLengths, textFeaturesLengths, emptyCount = analyzeTextLengths(dataFrame)
        trainingReady = analyzeDataQuality(dataFrame)
        
        #Generate summary
        generateSummary(dataFrame, outcomeCounts, guidelineCounts, fullTextLengths, trainingReady, emptyCount)
        
        print("Exploratory data analysis complete!")
        print(f"Visualizations saved in: {dataDir.parent / 'notebooks'}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
