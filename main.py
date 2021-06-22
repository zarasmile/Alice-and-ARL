import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain, combinations
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
from eclat import eclat


def powerset(iterable):
    s = list(iterable)
    yield from chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


class AnalyzerGen:
    def __init__(self):
        raise NotImplementedError


class AnalyzerBase(metaclass=ABCMeta):
    def __init__(self):
        self.support = {}
        self.rules = defaultdict(list)

    def _load_transactions(self, transactions, items):
        self.transactions = list(transactions)
        if not items:
            self.items = list({i for i in chain.from_iterable(transactions)})
        else:
            self.items = items
        return len(self.items), len(self.transactions)

    @abstractmethod
    def _calc_support(self, data, min_support=0.1):
        """support dict"""

    def gen_rules(self, min_confidence=0.5, min_lift=1.0):
        for itemset in self.support:
            for s in powerset(itemset):
                from_itemset = frozenset(s)
                to_itemset = itemset.difference(s)
                confidence = self.support[itemset] / self.support[from_itemset]
                lift = self.support[itemset] / (self.support[from_itemset] * self.support[to_itemset])
                if confidence >= min_confidence and lift >= min_lift:
                    self.rules[from_itemset].append({'to': to_itemset, 'confidence': confidence, 'lift': lift})
        return len(self.rules)

    def fit(self, transactions, items=None, min_support=0.1, min_confidence=0.5, min_lift=1.0):
        items_count, transactions_count = self._load_transactions(transactions, items)
        return {'items_count': items_count,
                'transactions_count': transactions_count,
                'support_count': self._calc_support(transactions, min_support),
                'rule_count': self.gen_rules(min_confidence=min_confidence, min_lift=min_lift)}

    def predict(self, itemset):
        itemset = frozenset(itemset)
        max_confidence = 0
        max_lift = 0
        max_rule = None
        if itemset in self.rules:
            for rule in self.rules[itemset]:
                if rule['confidence'] >= max_confidence and rule['lift'] >= max_lift:
                    max_confidence = rule['confidence']
                    max_lift = rule['lift']
                    max_rule = rule
        return max_rule


class FPGrowth(AnalyzerBase):
    def _calc_support(self, data, min_support=0.1):
        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        result = fpgrowth(df, min_support=min_support)
        support = {}
        for idx, row in result.iterrows():
            support[row['itemsets']] = row['support']
        self.support = support
        return len(self.support)


class Apriori(AnalyzerBase):
    def _calc_support(self, data, min_support=0.1):
        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        result = apriori(df, min_support=min_support)
        support = {}
        for idx, row in result.iterrows():
            support[row['itemsets']] = row['support']
        self.support = support
        return len(self.support)


class Eclat(AnalyzerBase):
    def _calc_support(self, data, min_support=0.1):
        result = eclat(data, min_support)
        support = {}
        for idx, row in result.iterrows():
            support[row['itemsets']] = row['support']
        self.support = support
        return len(self.support)


class FreqPM:
    analyzers = {'fp-growth': FPGrowth, 'apriori': Apriori, 'eclat': Eclat}

    @classmethod
    def model(cls, analyzer):
        return cls.analyzers.get(analyzer, AnalyzerGen)()


def read_data(data_path):
    dataset = []
    itemset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        file = csv.reader(f, delimiter=' ', quotechar='\r')
        for row in file:
            dataset.append(row)
            for word in row:
                itemset.append(word)
    return {
        'dataset': dataset,
        'itemset': list(set(itemset))
    }


def generate_graphics(path, min_support=0.6, min_confidence=0.1, min_lift=0.1, algorithms=None, compare=False):
    data = read_data(path)
    dataset = data['dataset']
    items = data['itemset']

    if algorithms is None:
        algorithms = ['apriori', 'eclat', 'fp-growth']

    if compare:
        min_support_experiment_list = [0.6, 0.7, 0.8, 0.85, 0.9]
    else:
        min_support_experiment_list = [min_support]

    duration = [[], [], []]
    for min_support_experiment in min_support_experiment_list:
        for idx, algorithm in enumerate(algorithms):
            start_time = time.time()
            result = FreqPM.model(algorithm)
            result.fit(dataset, items, min_support=min_support_experiment, min_confidence=min_confidence,
                       min_lift=min_lift)
            duration[idx].append(time.time() - start_time)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    legends = []
    for idx, algorithm in enumerate(algorithms):
        line, = plt.plot(min_support_experiment_list, duration[idx], marker='o', label=algorithm)
        legends.append(line)
    plt.legend(handles=list(legends))

    for dur in duration:
        for xy in zip(min_support_experiment_list, dur):
            ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')

    plt.xlabel('minimum support')
    plt.ylabel('execution time')
    plt.grid()
    fig.savefig('./data/graphic.jpeg')
