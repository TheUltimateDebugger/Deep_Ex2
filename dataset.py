import os
import torch
from torch.utils.data import Dataset


class PeptideDataset(Dataset):
    """
    The dataset of the Peptides. Reads the information from a folder
    and organises it in a way that a network can read it
    """
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.peptides = []
        self.labels = []
        # the amino acids:
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.label_mapping = {}
        self._load_data()

    def _load_data(self):
        """
        Loads the data from the files and makes the labels for them
        """
        allele_files = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')]
        current_label = 0
        for filename in sorted(allele_files):
            filepath = os.path.join(self.folder_path, filename)
            base_name = filename.replace('.txt', '')

            if base_name not in self.label_mapping:
                self.label_mapping[base_name] = current_label
                current_label += 1

            label = self.label_mapping[base_name]
            with open(filepath, 'r') as file:
                lines = file.read().splitlines()
                for line in lines:
                    if line.strip():
                        self.peptides.append(line.strip())
                        self.labels.append(label)

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: the peptide with a one hot vector with the Y value (the correct allele)
        """
        peptide = self.peptides[idx]
        label = self.labels[idx]

        one_hot = torch.zeros(len(peptide), len(self.amino_acids))
        for i, aa in enumerate(peptide):
            aa_idx = self.aa_to_idx.get(aa, None)
            if aa_idx is not None:
                one_hot[i, aa_idx] = 1.0
            else:
                raise ValueError(f"Unknown amino acid '{aa}' in peptide '{peptide}'")

        one_hot = one_hot.view(-1)
        return one_hot, torch.tensor(label, dtype=torch.long)
