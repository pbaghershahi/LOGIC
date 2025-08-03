import torch, random, os, ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *
from data_utils import *
from model import *
from eval_utils import test
import ipdb

logger = logging.getLogger("global_logger")


def pretrain_gnn(
    dataset,
    model_config,
    optimizer_config,
    training_config,
    eval_step = 1,
    save_model = True, 
    pretext_task = "classification",
    model_dir = "./pretrained",
):
    model = PretrainedModel(**model_config)
    model.to(model.device)

    if pretext_task == "classification":
        obj_fun = nn.CrossEntropyLoss()
    else:
        raise Exception("Pretext task is not implemented yet!")
    optimizer = Adam(model.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
    
    test_loss, test_acc, test_f1 = test(model, dataset, validation = False)
    logger.info(f'GNN Before Pretraining: -- Test Loss: {test_loss:.3f} -- Test ACC: {test_acc:.3f} -- Test F1-score: {test_f1:.3f}')

    train_ds = dataset.train_loader.dataset
    num_epochs = training_config["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        for i, batch_idxs in enumerate(dataset.train_loader):
            optimizer.zero_grad()
            temp_labels = train_ds.y[batch_idxs].to(model.device)
            scores, _ = model(
                batch = train_ds,
                decoder = True,
            )
            loss = obj_fun(
                scores[batch_idxs],
                temp_labels
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch % eval_step == 0 and epoch > 0:
            val_loss, val_acc, val_f1 = test(model, dataset, validation = True)
            logger.info(
                f"Epoch: {epoch}/{num_epochs} -- Train Loss: {loss:.3f} -- " +
                f"Validation Loss: {val_loss:.3f} -- Validation ACC: {val_acc:.3f} -- Validation F1: {val_f1:.3f}"
            )

    results = {}
    test_loss, results["pretrained_test_acc"], results["pretrained_test_f1"] = test(model, dataset, validation = False)
    logger.info(
        f"GNN After Pretraining: -- Train Loss: {loss:.3f} "
        f"-- Test Loss: {test_loss:.3f} "
        f"-- Test ACC: {results['pretrained_test_acc']:.3f} "
        f"-- Test F1: {results['pretrained_test_f1']:.3f}"
    )

    if save_model:
        model_path = os.path.join(model_dir, f"{model_config['gnn_type']}.pth")
        torch.save(
            {
                'model_state_dict': model.state_dict(),
            }, model_path
        )
        logger.info(f"Model saved to: {model_path}")
    else:
        model_path = "Won't be stored"
    return model, model_path, results



def context_loss(soft_prompts, keyword_embeddings):
    # Mean pool the soft prompt across tokens
    pooled_prompt = soft_prompts.mean(dim=1)
    pooled_prompt = F.normalize(pooled_prompt, dim=-1)
    keyword_embeddings = F.normalize(keyword_embeddings, dim=-1)
    # Cosine similarity loss (maximize similarity = minimize negative cosine)
    return - (pooled_prompt * keyword_embeddings).sum(dim=-1).mean()



def contrastive_loss_self_sim(embeddings, temperature=0.5):
    
    soft_prompts = soft_prompts.mean(dim=1)
    embeddings = F.normalize(embeddings, dim=-1)
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (batch_size, batch_size)

    labels = torch.arange(embeddings.size(0)).to(embeddings.device)
    loss = F.cross_entropy(similarity_matrix / temperature, labels)
    return loss

    

def contrastive_loss(embeddings, temperature=0.5):
    
    soft_prompts = soft_prompts.mean(dim=1)
    batch_size = embeddings.size(0)
    embeddings = F.normalize(embeddings, dim=-1)

    sim_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)
    sim_matrix = sim_matrix / temperature  # Scale by temperature

    # Mask self-similarity by assigning large negative value
    logits_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    logits = sim_matrix.masked_fill(~logits_mask, float('-inf'))

    logits = logits.clamp(min=-100)

    labels = torch.arange(batch_size, device=embeddings.device)

    # Cross-entropy loss
    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(log_probs, labels)
    return loss



def contrastive_distance_loss(soft_prompts, gnn_embeddings, temperature=0.1):
    soft_prompts = soft_prompts.mean(dim=1)
    soft_prompts = F.normalize(soft_prompts, dim=-1)
    gnn_embeddings = F.normalize(gnn_embeddings, dim=-1)

    soft_sim = torch.matmul(soft_prompts, soft_prompts.T) / temperature
    gnn_sim = torch.matmul(gnn_embeddings, gnn_embeddings.T) / temperature

    loss = F.mse_loss(soft_sim, gnn_sim)
    return loss



def contrastive_loss_from_gnn(soft_prompts, gnn_embeddings, temperature=0.5):
    soft_prompts = soft_prompts.mean(dim=1)
    soft_prompts = F.normalize(soft_prompts, dim=-1)
    gnn_embeddings = F.normalize(gnn_embeddings, dim=-1)

    # Compute similarity matrices
    soft_sim = torch.matmul(soft_prompts, soft_prompts.T)  # (B, B)
    gnn_sim = torch.matmul(gnn_embeddings, gnn_embeddings.T)  # (B, B)
    gnn_sim = gnn_sim / temperature
    gnn_sim_probs = F.softmax(gnn_sim, dim=1)

    # Cross-entropy between soft similarities and GNN similarity "labels"
    loss = -torch.sum(gnn_sim_probs * F.log_softmax(soft_sim, dim=1), dim=1).mean()
    return loss



def pretrain_projector(
    dataset,
    embed_func,
    tokenizer,
    gnn_embeds,
    projector_config,
    optimizer_config,
    training_config,
    gnn_preds = None,
    use_bow = False,
    eval_step = 50,
    model_dir = None, 
):

    projector = GNNtoSoftPrompt(**projector_config)
    projector.to(gnn_embeds.device)

    optimizer = Adam(projector.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])

    llm_emebds = []

    for i in range(len(dataset)):
        prompt_content = ""
        if use_bow:
            prompt_content += ", ".join(dataset._data.words[i])
            prompt_content += ", " + dataset._data.label_info[str(gnn_preds[i].item())]
        else:
            prompt_content = dataset._data.raw_text[i]
        
        tokenized_prompt = tokenizer(prompt_content, return_tensors="pt", add_special_tokens=False)

        prompt_embeds = embed_func(
            tokenized_prompt["input_ids"].to(torch.long).to(gnn_embeds.device)
        ).squeeze(0)
        prompt_embeds = prompt_embeds.mean(dim=0)
        llm_emebds.append(prompt_embeds)

    llm_emebds = torch.stack(llm_emebds, dim=0)

    gnn_embeds = F.normalize(gnn_embeds, dim=-1).to(gnn_embeds.device)
    llm_emebds = F.normalize(llm_emebds, dim=-1).to(gnn_embeds.device).detach()

    num_epochs = training_config["num_epochs"]
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        soft_prompts = projector(gnn_embeds)
        contrastive = contrastive_loss_from_gnn(soft_prompts, gnn_embeds, temperature=training_config["temperature"])
        context = context_loss(soft_prompts, llm_emebds)
        
        loss = (1 - training_config["beta"]) * contrastive + training_config["beta"] * context
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % eval_step == 0:
            logger.info(f"Step {i}: Total Loss = {loss.item():.4f} | Contrastive = {contrastive.item():.4f} | Context = {context.item():.4f}")

    if model_dir:
        model_path = os.path.join(model_dir, "projector.pth")
        torch.save(
            {
                'model_state_dict': projector.state_dict(),
            }, model_path
        )
        logger.info(f"Model saved to: {model_path}")

    return projector, model_path