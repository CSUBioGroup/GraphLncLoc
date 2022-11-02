from models.lncRNA_lib import *

mapLocation = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
GraphLncLoc = []
for i in range(1, 6):
    GraphLncLoc.append(
        lncRNALocalizer(weightPath=f"checkpoints/Final_model/fold{i}_5.pkl", n_classes=5, map_location=mapLocation))


def lncRNA_loc_predict(lncRNA):
    return vote_predict(GraphLncLoc, lncRNA.upper())


if __name__ == "__main__":
    sequence = "ACCAUUUUCAUAGAAUCAUUAUGAUGAUUAACUAAAUUGUGGAUAAAUAAAAACCAAAAUACCAGUGGAAUGAAAUAUUGUCAUUACUUGGCCAUGAGCUUUUAAGUGUGUGUGUCUAUGUUUCUAGCGGGGCACAUGCCUGUAAUCAUAGCUCUUGGUAGGAAGUGACAAGAUUAGGCCUUGAGUAGAGCAAGUUUGAGGCUAAUAUGGGCUGUGUGAGACACUAUCUCAAAGCAAACAGUAUGUUCCUGAGAUAGGGCUGUAACUCAAUGACAAAGCACUGUCUAAUGUGCGAAGCCCCCCAAGUUCAAUCCACAGCCACUCAGAGGCUCCAGAGUCCUCCCUUCUACUCCUAUUGAGUACAAGUUUCAUUAGAGGUGAUACACACCAUUUUUCCAAAUGGCUAAUUGUGCCUUUUACAUAUAGUUGAUGAUAAAAAUACAUAUUGUGAAAUAAAAUAUUGUAUAUAUGUGUGCAUGCACACAUCUACUGUUCUGAACAGAGAACCCAGGACUUUAUGAAGGACUUUAUUAACUAUUGCCCCUAAGCCAUGAAAGAUAUUUCUUUGACUGGCAUAUUCGCAGCAUGAUUUUUCCACUCUUCUGCACUCUUCAACUUGUGGUUCGUGUAUGCAUGUGUGUACAAACACACAUACACAUGCAGUGUAGAGGUGUCACUAUUUUGACCAGGCUCGUCU"
    print(lncRNA_loc_predict(sequence))
