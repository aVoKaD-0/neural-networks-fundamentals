import torch

class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = int(vocab_size)
        self.depth = max(1, (self.vocab_size - 1).bit_length())
        self.max_path_length = self.depth
        self.num_internal_nodes = 2 ** self.depth - 1

    def targets_for_index(self, word_index):
        return format(int(word_index), f'0{self.depth}b')

    def node_id_from_prefix(self, prefix_bits):
        if prefix_bits == '':
            return 0
        return (2 ** len(prefix_bits) - 1) + int(prefix_bits, 2)

    def path_and_targets(self, word_index):
        binary_targets = self.targets_for_index(word_index)
        path = []
        targets = []
        for step in range(self.depth):
            prefix = binary_targets[:step]
            path.append(self.node_id_from_prefix(prefix))
            targets.append(int(binary_targets[step]))
        return path, targets

    def __call__(self, context_word):
        device = context_word.device
        word_indices = context_word.detach().cpu().view(-1).tolist()

        paths = []
        targets = []
        for word_index in word_indices:
            path, target = self.path_and_targets(int(word_index))
            paths.append(path)
            targets.append(target)

        path = torch.tensor(paths, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        mask = torch.ones(len(word_indices), self.max_path_length, dtype=torch.float32, device=device)

        return {'path': path, 'targets': targets, 'mask': mask}

class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.targets = BinaryIndexTree(vocab_size)
        self.num_internal_nodes = self.targets.num_internal_nodes
        self.max_path_length = self.targets.max_path_length
        self.decoder = torch.nn.Embedding(self.num_internal_nodes, self.embedding_dim)
        torch.nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)

    def forward(self, embedding, target_word):
        tree_output = self.targets(target_word)
        path = tree_output['path']
        targets = tree_output['targets']
        mask = tree_output['mask']

        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        probabilities = torch.sigmoid(logits)
        target_probabilities = torch.where(targets.bool(), probabilities, 1.0 - probabilities)
        total_probability = target_probabilities.prod(dim=1)

        per_node_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        per_node_loss = per_node_loss * mask
        per_word_loss = per_node_loss.sum(dim=1)
        loss = per_word_loss.mean()

        return {
            'path': path,
            'targets': targets,
            'mask': mask,
            'logits': logits,
            'probabilities': probabilities,
            'target_probabilities': target_probabilities,
            'total_probability': total_probability,
            'per_node_loss': per_node_loss,
            'per_word_loss': per_word_loss,
            'loss': loss,
        }

class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        self.hierarchical_softmax = HierarchicalSoftmax(self.embedding_dim, self.vocab_size)
        self.decoder = self.hierarchical_softmax.decoder
        self.num_internal_nodes = self.hierarchical_softmax.num_internal_nodes

    def forward(self, batch):
        center_word = batch['data']['center_word']
        embedding = self.encoder(center_word)

        batch['signals'] = {'embedding': embedding}
        batch['postprocessed'] = {}

        context_word = batch['data'].get('context_word')
        if context_word is not None:
            hsoftmax_output = self.hierarchical_softmax(embedding, context_word)

            batch['data']['path'] = hsoftmax_output['path']
            batch['data']['targets'] = hsoftmax_output['targets']
            batch['data']['mask'] = hsoftmax_output['mask']

            batch['signals']['logits'] = hsoftmax_output['logits']
            batch['signals']['probabilities'] = hsoftmax_output['probabilities']
            batch['signals']['target_probabilities'] = hsoftmax_output['target_probabilities']
            batch['signals']['total_probability'] = hsoftmax_output['total_probability']
            batch['signals']['loss'] = hsoftmax_output['loss']

            batch['postprocessed']['targets'] = (hsoftmax_output['probabilities'] >= 0.5).to(torch.long)

        return batch