import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import lightning as L
from typing import Dict, Any
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GlobalAttention, LayerNorm
from src.datas.samples import XFGBatch
from src.vocabulary import Vocabulary
from src.metrics import Statistic


class MemoryEfficientSTEncoder(torch.nn.Module):
    """Memory-efficient Statement Encoder for GTX 1650"""
    
    def __init__(self, config, vocab, vocabulary_size, pad_idx):
        super().__init__()
        self.__pad_idx = pad_idx
        self.config = config
        
        # Standard embedding (no positional encoding to save memory)
        self.embedding = nn.Embedding(
            vocabulary_size, 
            config.embed_size, 
            padding_idx=pad_idx
        )
        
        # Memory-efficient LSTM (2 layers max)
        self.rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.rnn.hidden_size,
            num_layers=config.rnn.num_layers,  # 2 layers
            bidirectional=config.rnn.use_bi,
            dropout=config.rnn.drop_out if config.rnn.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Calculate correct dimensions for attention
        rnn_output_size = config.rnn.hidden_size * (2 if config.rnn.use_bi else 1)  # 256 * 2 = 512
        
        # 🔥 FIX: Ensure embed_dim is divisible by num_heads
        # 512 is divisible by 8, 4, 2, 1
        # Let's use 8 heads for 512 dim (512 / 8 = 64 per head)
        num_heads = 8 if rnn_output_size >= 512 else 4
        
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_output_size,  # 512
            num_heads=num_heads,        # 8 (512/8 = 64 per head)
            dropout=0.1,
            batch_first=True
        )
        
        # Simple output projection
        self.output_proj = nn.Linear(rnn_output_size, config.rnn.hidden_size)
        self.layer_norm = nn.LayerNorm(config.rnn.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq):
        # Standard forward pass with checkpointing
        # Only checkpoint the most memory-intensive parts
        if self.training and self.config.get('use_gradient_checkpointing', False):
            # Checkpoint only the LSTM + attention part (most memory intensive)
            return checkpoint(self._forward_with_checkpoint, seq, use_reentrant=False)
        else:
            return self._forward_impl(seq)
    
    def _forward_with_checkpoint(self, seq):
        """Forward pass with selective checkpointing"""
        # Embedding (lightweight, no checkpointing)
        embedded = self.embedding(seq)
        
        # Create attention mask
        mask = (seq == self.__pad_idx)
        
        # LSTM + Attention (memory intensive, checkpoint this part)
        def lstm_attention_forward(embedded, mask):
            rnn_output, _ = self.rnn(embedded)
            
            # Single attention layer
            if not mask.all():
                attended_output, _ = self.attention(
                    rnn_output, rnn_output, rnn_output,
                    key_padding_mask=mask
                )
                
                # Mean pooling with mask
                lengths = (~mask).sum(dim=1, keepdim=True).float()
                attended_output = attended_output.masked_fill(mask.unsqueeze(-1), 0)
                pooled = attended_output.sum(dim=1) / lengths.clamp(min=1)
            else:
                pooled = rnn_output.mean(dim=1)
            
            return pooled
        
        # Apply checkpointing only to the heavy computation
        pooled = checkpoint(lstm_attention_forward, embedded, mask, use_reentrant=False)
        
        # Output projection (lightweight, no checkpointing)
        output = self.output_proj(pooled)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output

    def _forward_impl(self, seq):
        """Standard forward pass without checkpointing"""
        # Embedding
        embedded = self.embedding(seq)
        
        # Create attention mask
        mask = (seq == self.__pad_idx)
        
        # LSTM processing
        rnn_output, _ = self.rnn(embedded)
        
        # Single attention layer
        if not mask.all():
            attended_output, _ = self.attention(
                rnn_output, rnn_output, rnn_output,
                key_padding_mask=mask
            )
            
            # Mean pooling with mask
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            attended_output = attended_output.masked_fill(mask.unsqueeze(-1), 0)
            pooled = attended_output.sum(dim=1) / lengths.clamp(min=1)
        else:
            pooled = rnn_output.mean(dim=1)
        
        # Output projection
        output = self.output_proj(pooled)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class MemoryEfficientGraphEncoder(torch.nn.Module):
    """Memory-efficient Graph Encoder for GTX 1650"""
    
    def __init__(self, config, vocab, vocabulary_size, pad_idx):
        super().__init__()
        self.config = config
        self.st_embedding = MemoryEfficientSTEncoder(config, vocab, vocabulary_size, pad_idx)
        
        # Moderate hidden size for memory efficiency
        self.hidden_size = config.hidden_size  # 320
        
        # Input GCN
        self.input_gcn = GCNConv(config.rnn.hidden_size, self.hidden_size)
        
        # Memory-efficient GCN layers (4 layers max)
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(config.n_hidden_layers):  # 4 layers
            self.gcn_layers.append(GCNConv(self.hidden_size, self.hidden_size))
            self.layer_norms.append(LayerNorm(self.hidden_size))
            self.dropouts.append(nn.Dropout(config.drop_out))
        
        # Memory-efficient global pooling (single method)
        self.global_pool = GlobalAttention(
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 1)
            )
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(config.rnn.hidden_size, self.hidden_size)

    def forward(self, batched_graph):
        # Only checkpoint the GCN layers (most memory intensive)
        if self.training and self.config.get('use_gradient_checkpointing', False):
            return checkpoint(self._forward_with_checkpoint, batched_graph, use_reentrant=False)
        else:
            return self._forward_impl(batched_graph)
    
    def _forward_with_checkpoint(self, batched_graph):
        """Forward pass with selective checkpointing for GCN layers"""
        # Statement encoding (lightweight)
        node_embedding = self.st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        
        # Input layer with residual (lightweight)
        x = F.relu(self.input_gcn(node_embedding, edge_index))
        residual = self.residual_proj(node_embedding)
        x = x + residual
        
        # Checkpoint only the GCN layers (memory intensive)
        def gcn_forward(x, edge_index):
            for i, (gcn, norm, dropout) in enumerate(zip(self.gcn_layers, self.layer_norms, self.dropouts)):
                residual = x
                x = gcn(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = dropout(x)
                x = x + residual  # Residual connection
            return x
        
        # Apply checkpointing only to GCN layers
        x = checkpoint(gcn_forward, x, edge_index, use_reentrant=False)
        
        # Global pooling (lightweight)
        graph_repr = self.global_pool(x, batch)
        
        return graph_repr

    def _forward_impl(self, batched_graph):
        """Standard forward pass without checkpointing"""
        # Statement encoding
        node_embedding = self.st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        
        # Input layer with residual
        x = F.relu(self.input_gcn(node_embedding, edge_index))
        residual = self.residual_proj(node_embedding)
        x = x + residual
        
        # Hidden layers with residual connections
        for i, (gcn, norm, dropout) in enumerate(zip(self.gcn_layers, self.layer_norms, self.dropouts)):
            residual = x
            x = gcn(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            x = x + residual  # Residual connection
        
        # Global pooling
        graph_repr = self.global_pool(x, batch)
        
        return graph_repr


class MemoryEfficientClassifier(torch.nn.Module):
    """Memory-efficient Classifier for GTX 1650"""
    
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        
        # Progressive size reduction (memory-efficient)
        layer_sizes = [input_size, 512, 384, 256]  # Moderate progression
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(input_size if i == 0 else layer_sizes[i], layer_sizes[i + 1]))
            self.layer_norms.append(nn.LayerNorm(layer_sizes[i + 1]))
            
            # Progressive dropout
            dropout_rate = 0.3 + (i * 0.1)
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Final classifier
        self.classifier = nn.Linear(256, 2)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Classifier is lightweight, no need for checkpointing
        return self._forward_impl(x)
    
    def _forward_impl(self, x):
        # Progressive processing
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = dropout(x)
        
        # Final prediction with temperature scaling
        logits = self.classifier(x)
        calibrated_logits = logits / self.temperature
        
        return calibrated_logits


class EnhancedLossFunction(nn.Module):
    """Enhanced loss function optimized for UAV imbalanced data"""
    
    def __init__(self, config):
        super().__init__()
        
        # Enhanced focal loss parameters (optimized for UAV)
        self.focal_alpha = config.hyper_parameters.focal_alpha  # 0.85
        self.focal_gamma = config.hyper_parameters.focal_gamma  # 4.0
        
        # Class weights for UAV imbalanced data
        self.register_buffer('class_weights', torch.tensor([0.35, 2.85]))
        
        # Label smoothing
        self.label_smoothing = config.hyper_parameters.get('label_smoothing', 0.1)

    def focal_loss(self, logits, labels):
        """Enhanced focal loss for severe class imbalance"""
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def label_smoothing_loss(self, logits, labels):
        """Label smoothing for better generalization"""
        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = F.nll_loss(log_probs, labels, reduction='none')
        smooth_loss = -log_probs.mean(dim=1)
        
        return (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

    def forward(self, logits, labels):
        # Combined loss: 80% focal + 20% label smoothing
        focal = self.focal_loss(logits, labels)
        smooth = self.label_smoothing_loss(logits, labels).mean()
        
        return 0.8 * focal + 0.2 * smooth


class MemoryEfficientDeepWukong(L.LightningModule):
    """Memory-efficient DeepWukong optimized for GTX 1650 4GB"""
    
    def __init__(self, config, vocab, vocabulary_size, pad_idx):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        
        # Memory-efficient components
        self.graph_encoder = MemoryEfficientGraphEncoder(config.gnn, vocab, vocabulary_size, pad_idx)
        self.classifier = MemoryEfficientClassifier(config.gnn.hidden_size, config.classifier)
        
        # Enhanced loss function
        self.loss_fn = EnhancedLossFunction(config)
        
        # Mixed precision settings
        self.use_mixed_precision = config.hyper_parameters.get('use_mixed_precision', False)
        if self.use_mixed_precision:
            self.automatic_optimization = False

    def configure_optimizers(self):
        """Memory-efficient optimizer configuration"""
        # Different learning rates for different components
        encoder_params = list(self.graph_encoder.parameters())
        classifier_params = list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self._config.hyper_parameters.learning_rate * 0.8},
            {'params': classifier_params, 'lr': self._config.hyper_parameters.learning_rate}
        ], weight_decay=self._config.hyper_parameters.weight_decay)
        
        # Learning rate scheduler
        if self._config.hyper_parameters.get('use_lr_scheduler', False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',
                patience=self._config.hyper_parameters.get('patience_lr', 8),
                factor=self._config.hyper_parameters.get('factor_lr', 0.7),
                verbose=True
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1",
                    "frequency": 1
                }
            }
        
        return optimizer

    def forward(self, batch):
        """Memory-efficient forward pass"""
        # Clear cache periodically to prevent memory buildup
        if self.training and hasattr(self, 'global_step') and self.global_step % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Graph encoding
        graph_repr = self.graph_encoder(batch)
        
        # Classification
        logits = self.classifier(graph_repr)
        
        return logits

    def training_step(self, batch, batch_idx):
        """Memory-efficient training step"""
        if self.use_mixed_precision:
            return self._mixed_precision_training_step(batch, batch_idx)
        else:
            return self._standard_training_step(batch, batch_idx)
    
    def _standard_training_step(self, batch, batch_idx):
        logits = self(batch.graphs)
        loss = self.loss_fn(logits, batch.labels)
        
        # Metrics calculation
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(batch.labels, preds, 2)
            metrics = statistic.calculate_metrics(group="train")
            
            batch_size = batch.labels.size(0)
            self.log_dict(metrics, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        
        return {"loss": loss, "statistic": statistic}
    
    def _mixed_precision_training_step(self, batch, batch_idx):
        """Mixed precision training for memory efficiency"""
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            logits = self(batch.graphs)
            loss = self.loss_fn(logits, batch.labels)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self._config.hyper_parameters.gradient_accumulation_steps
        
        # Backward pass with scaler
        self.manual_backward(scaled_loss)
        
        # Gradient accumulation
        if (batch_idx + 1) % self._config.hyper_parameters.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(batch.labels, preds, 2)
            metrics = statistic.calculate_metrics(group="train")
            
            batch_size = batch.labels.size(0)
            self.log_dict(metrics, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch, batch_idx):
        """Memory-efficient validation step"""
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            logits = self(batch.graphs)
            loss = self.loss_fn(logits, batch.labels)

        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(batch.labels, preds, 2)
            metrics = statistic.calculate_metrics(group="val")
            
            batch_size = batch.labels.size(0)
            self.log("val_loss", loss, on_step=False, on_epoch=True, 
                    prog_bar=True, batch_size=batch_size)
            self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch, batch_idx):
        """Memory-efficient test step"""
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            logits = self(batch.graphs)
            loss = self.loss_fn(logits, batch.labels)

        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(batch.labels, preds, 2)
            metrics = statistic.calculate_metrics(group="test")
            
            batch_size = batch.labels.size(0)
            self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        return {"loss": loss, "statistic": statistic}

    def on_train_epoch_end(self):
        """Clear cache at end of epoch"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """Clear cache after validation"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()