import os, logging
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parents[0])
sys.path.insert(0, ROOT_PATH)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from datetime import datetime
import functools
import argparse
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import shutil
import json, yaml
import itertools
import networkx as nx
import re
from tqdm import tqdm
from torch_geometric.utils import to_networkx, subgraph
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer, GPT2Model
import transformers
import gc
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import nltk
import pickle
from math import ceil
import torch.nn.functional as F
from networkx.algorithms.approximation import steiner_tree
from huggingface_hub.hf_api import HfFolder
from TAGLAS import get_dataset
from datetime import datetime
from train_utils import *
from eval_utils import *
from utils import *
from model import PretrainedModel, GNNToSoftPrompt


hugging_face_token = os.environ.get('HUGGINGFACE_TOKEN')
HfFolder.save_token(hugging_face_token)



def main(args):

    data_root_dir = os.path.join(ROOT_PATH, "data")
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

    if args.not_verbose:
        global_logger = DummyLogger()
        eval_logger = DummyLogger()
    else:
        if args.log_dir is None:
            log_dir = Path(ROOT_PATH) / "log/"
        else:
            log_dir = Path(args.log_dir) / "LOGIC/log"

        general_log_dir = Path(log_dir) / "general"
        evaluations_log_dir = Path(log_dir) / "evaluation"
        os.makedirs(general_log_dir, exist_ok=True)
        os.makedirs(evaluations_log_dir, exist_ok=True)

        log_file_paths = [
            general_log_dir / f"global_{args.method}_{args.dataset}_{exec_name}.log", 
            evaluations_log_dir / f"eval_{args.method}_{args.dataset}_{exec_name}.log"
        ]
        global_logger = setup_logger(
            name="global_logger", 
            level=logging.INFO, 
            log_file=log_file_paths[0],
            stream_handler=True,
        )
        eval_logger = setup_logger(
            name="eval_logger", 
            level=logging.INFO, 
            log_file=log_file_paths[1],
            stream_handler=True,
            formatter=logging.Formatter(fmt='%(message)s')
        )
        global_logger.info(f"Logging to: {log_file_paths}")

    if args.config_from_file != "":
        global_logger.info(f"Reading config from: {args.config_from_file}")
        with open(args.config_from_file, 'r') as infile:
            all_args = vars(args)
            input_args = []
            for key, value in all_args.items():
                if value is not None:
                    input_args.append(key)
            file_args = yaml.safe_load(infile)
            args = {key:file_args[key] if (key in file_args and key not in input_args) else value for key, value in all_args.items()}
            args = argparse.Namespace(**args)
    
    arg_seeds = np.random.randint(1000, 5000, (args.total_iters,)) if len(args.seed) == 0 else args.seed
    total_iters = len(arg_seeds)
    global_logger.info(args)
    if args.write_new_output:
        output_dir = os.path.join(
            ROOT_PATH,
            "output",
            f"{args.dataset}-{args.llm_model}-{exec_name}/"
        )
        model_dir = os.path.join(
            ROOT_PATH,
            "pretrained",
            f"{args.dataset}-{args.llm_model}-{exec_name}/"
        )
    else:
        output_dir = os.path.join(
            ROOT_PATH,
            "output",
            f"{args.dataset}-{args.llm_model}/"
        )
        model_dir = os.path.join(
            ROOT_PATH,
            "pretrained",
            f"{args.dataset}-{args.llm_model}/"
        )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    global_logger.info(f"Seeds: {arg_seeds}")
    global_logger.info(f"Saving outputs to:\n{output_dir}")
    global_logger.info("#"*100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.download_nltk_stopwords:
        nltk.download('stopwords')

    for i in range(total_iters):
        global_logger.info(f"Started round {i+1}/{total_iters} of experiments!")
        fix_seed(arg_seeds[i])

        if args.dataset == "products":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        elif args.dataset == "cora":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        elif args.dataset == "wikics":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        elif args.dataset == "liar":
            dataset = get_node_dataset(
                dataset_name = args.dataset,
                train_test_split = args.train_test_split,
                batch_size = args.gnn_batch_size,
                normal_mode = args.normal_mode,
                seed = arg_seeds[i],
                sample_size = args.dataset_sample_size,
                random_sampling = args.dataset_random_sampling,
                num_workers = args.gnn_dataloader_num_workers,
                use_bow = args.use_bow,
                num_hops = args.neighborhood_max_hops,
                max_num_frequent_words = args.max_num_frequent_words
            )
        else:
            raise NotImplementedError

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_gnn_paths = args.pretrained_gnn_path if len(args.pretrained_gnn_path) > 0 else []
        processed_data_paths = args.processed_data_path if len(args.processed_data_path) > 0 else []
        pretrained_proj_paths = args.pretrained_projector_path if len(args.pretrained_projector_path) > 0 else []
        all_results = {}

        model_config = dict(
            gnn_type = args.gnn_type,
            in_channels = dataset.n_feats,
            hidden_channels = args.gnn_h_dim,
            out_channels = dataset.num_classes,
            gnn_num_hid_layers = args.gnn_num_hid_layers, 
            dropout = args.gnn_dropout,
            with_bn = False,
            decoder_type = args.gnn_decoder_type,
            device = device,
            with_last_dropout = args.gnn_with_last_dropout,
            mode = args.gnn_task_mode
        )
        optimizer_config = dict(
            lr = args.gnn_lr,
            scheduler_step_size = args.gnn_step_size,
            scheduler_gamma = args.gnn_gamma,
            weight_decay = args.gnn_weight_decay
        )
        training_config = dict(
            num_epochs = args.gnn_num_epochs
        )

        if len(pretrained_gnn_paths) == 0:
            global_logger.info(f"Pretraining {model_config['gnn_type']} on {args.dataset} started for {args.gnn_num_epochs} epochs")
            gnn, p_path, p_results = pretrain_gnn(
                dataset,
                model_config,
                optimizer_config,
                training_config,
                eval_step = args.gnn_eval_step,
                save_model = True,
                pretext_task = "classification",
                model_dir = model_dir,
            )
            pretrained_gnn_paths.append(p_path)
        else:
            p_results = {}
            gnn = PretrainedModel(**model_config).to(device)
            load_model(gnn, read_checkpoint=True, pretrained_path=pretrained_gnn_paths[i])
            _, p_results["pretrained_test_acc"], p_results["pretrained_test_f1"] = test(gnn, dataset, validation = False)
            global_logger.info(
                f"Reading GNN model from: {pretrained_gnn_paths[i]}\n"
                f"GNN After Pretraining: "
                f"-- Test ACC: {p_results['pretrained_test_acc']:.3f} "
                f"-- Test F1: {p_results['pretrained_test_f1']:.3f}"
            )

        gnn.eval()

        if len(processed_data_paths) == 0:
            with torch.no_grad():
                gnn_logits, gnn_embeds = gnn(batch=dataset._data)
                gnn_preds = gnn_logits.argmax(dim=1)
            outdict = dict(
                dataset = dataset,
                gnn_embeds = gnn_embeds.detach().to("cpu"),
                gnn_logits = gnn_logits.detach().to("cpu"),
                gnn_preds = gnn_preds.detach().to("cpu")
            )
            path_to_saved_data = os.path.join(output_dir, "data.pth")
            torch.save(outdict, path_to_saved_data)
            global_logger.info(f"Data and GNN preds saved to {path_to_saved_data}")
            processed_data_paths.append(path_to_saved_data)
        else:
            global_logger.info(f"Reading Data and GNN preds from: {processed_data_paths[i]}")
            output_dict = torch.load(processed_data_paths[i])
            dataset = output_dict["dataset"]
            gnn_preds, gnn_embeds, gnn_logits = output_dict["gnn_preds"].to(device), output_dict["gnn_embeds"].to(device), output_dict["gnn_logits"].to(device)

        eval_config = dict(
            max_num_eval_nodes = args.max_num_eval_nodes,
            llm_node_batch_size = args.llm_node_batch_size,
            save_every = args.save_generation_every
        )

        proj_embeds = None
        
        if args.method == "logic":

            tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            llm = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype = torch.bfloat16,
                device_map = "auto"
            )

            llm_type = args.llm_model.split("/")[0]
            if llm_type in ["meta-llama", "mistralai", "microsoft"]:
                embed_func = llm.model.embed_tokens
                with_chat_template = True
                eval_config["max_position_embeddings"] = llm.config.max_position_embeddings
            elif llm_type in ["openai"]:
                embed_func = llm.transformer.wte
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.config.n_positions
            elif llm_type in ["EleutherAI"]:
                embed_func = llm.transformer.wte
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.config.max_position_embeddings

            projector_config = dict(
                gnn_h_dim = args.gnn_h_dim, 
                num_tokens = args.num_tokens, 
                llm_h_dim = llm.config.hidden_size,
                gnn_out_dim = dataset.num_classes if args.proj_with_backward else None
            )
            optimizer_config = dict(
                lr = args.projector_lr,
                scheduler_step_size = args.projector_step_size,
                scheduler_gamma = args.projector_gamma,
                weight_decay = args.projector_weight_decay
            )
            training_config = dict(
                num_epochs = args.projector_num_epochs,
                temperature = args.contrastive_temperature,
                proj_contrastive_w = args.projector_contrastive_w,
                proj_context_w = args.projector_context_w,
                proj_mutualinfo_w = args.projector_mutualinfo_w
            )

            if len(pretrained_proj_paths) == 0:
                projector, p_path = pretrain_projector(
                    dataset,
                    embed_func,
                    tokenizer,
                    gnn_embeds,
                    gnn_logits,
                    projector_config,
                    optimizer_config,
                    training_config,
                    gnn_preds = gnn_preds,
                    use_bow = args.use_bow,
                    eval_step = 50,
                    model_dir = model_dir,
                )
                pretrained_proj_paths.append(p_path)
            else:
                global_logger.info(f"Reading projector model from: {pretrained_proj_paths[i]}")
                projector = GNNToSoftPrompt(**projector_config).to(device)
                load_model(projector, read_checkpoint=True, pretrained_path=pretrained_proj_paths[i])
            
            projector.eval()
            proj_embeds, _ = projector(gnn_embeds)
            proj_embeds = proj_embeds.to(dtype=torch.bfloat16)

            generated_exps = generate_exp_by_llm(
                dataset = dataset,
                model = llm,
                tokenizer = tokenizer,
                gnn_embeds = proj_embeds,
                gnn_preds = gnn_preds,
                eval_config = eval_config,
                embed_func = embed_func,
                generate_by = "embedding",
                save_to = os.path.join(output_dir, "LOGIC_explanations.pt"),
                save_every = 20,
                with_chat_template = with_chat_template
            )

            eval_llm_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                generated_exps = generated_exps,
                generated_with_pipeline = False,
                num_eval_samples = args.max_num_eval_nodes
            )
        
        elif args.method == "llm":

            tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
            llm = transformers.pipeline(
                "text-generation",
                model=args.llm_model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                token = key
            )

            llm_type = args.llm_model.split("/")[0]
            if llm_type in ["meta-llama", "mistralai", "microsoft"]:
                with_chat_template = True
                eval_config["max_position_embeddings"] = llm.model.config.max_position_embeddings
            elif llm_type in ["openai"]:
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.model.config.n_positions
            elif llm_type in ["EleutherAI"]:
                with_chat_template = False
                eval_config["max_position_embeddings"] = llm.model.config.max_position_embeddings

            generated_exps = generate_exp_by_llm(
                dataset = dataset,
                model = llm,
                tokenizer = tokenizer,
                gnn_preds = gnn_preds,
                eval_config = eval_config,
                generate_by = "tokens",
                save_to = os.path.join(output_dir, "LLM_explanations.pkl"),
                save_every = 20,
                with_chat_template = with_chat_template
            )

            eval_llm_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                generated_exps = generated_exps,
                generated_with_pipeline = True,
                num_eval_samples = args.max_num_eval_nodes
            )

        elif args.method == "random":

            eval_gnn_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                method = args.method,
                num_eval_samples = args.max_num_eval_nodes
            )

        elif args.method == "node":

            eval_gnn_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                method = args.method,
                num_eval_samples = args.max_num_eval_nodes
            )

        elif args.method == "gnnexplainer":

            eval_gnnexplainer_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                num_eval_samples = args.max_num_eval_nodes,
                save_to = os.path.join(output_dir, "GNNExplainer_explanations.pt"),
            )

        elif args.method == "pgexplainer":

            eval_pgexplainer_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                num_eval_samples = args.max_num_eval_nodes,
                num_explainer_epochs = 30
            )

        elif args.method == "tage":

            exp_config = dict(
                gnn_h_dim = args.gnn_h_dim
            )
            eval_tage_explanations(
                dataset = dataset, 
                gnn = gnn,
                gnn_preds = gnn_preds,
                exp_config = exp_config,
                num_eval_samples = args.max_num_eval_nodes,
                num_explainer_epochs = 200,
            )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str)
    parser.add_argument('-lm', '--llm-model', type=str, help=(
        "Choose from: "
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
        "meta-llama/Llama-2-7b-chat-hf"
        "openai-community/gpt2"
        "mistralai/Mistral-7B-Instruct-v0.2"
        "EleutherAI/gpt-neo-2.7B"
        "microsoft/Phi-3-mini-4k-instruct"
        )
    )
    parser.add_argument('-dns', '--download-nltk-stopwords', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("-ub", "--use-bow", action='store_true')
    parser.add_argument("-gdnw", "--gnn-dataloader-num-workers", type=int, help="Number of graph dataloader workers")
    parser.add_argument("-dss", "--dataset-sample-size", type=int, help="Graph dataset sample size")
    parser.add_argument("-drs", "--dataset-random-sampling", action="store_true")
    parser.add_argument("-pgp", "--pretrained-gnn-path", nargs='*', default=[], type=str, help="Paths to the pretrained model")
    parser.add_argument("-pdp", "--processed-data-path", nargs='*', default=[], type=str, help="Paths to the processed data")
    parser.add_argument("-ppp", "--pretrained-projector-path", nargs='*', default=[], type=str, help="Paths to the pretrained model")
    parser.add_argument('-gwlp', '--gnn-with-last-dropout', action='store_true')
    parser.add_argument("-gtm", "--gnn-task-mode", type=str)
    parser.add_argument("-gwd", "--gnn-weight-decay", type=float, help="Rate of regularization")
    parser.add_argument("-gt", "--gnn-type", type=str, help="Type of base GNN: [gcn, gat, gin, sage]")
    parser.add_argument("-gdt", "--gnn-decoder-type", type=str, help="GNN decoder: [linear, gnn]")
    parser.add_argument("-gnh", "--gnn-num-hid-layers", type=int, help="Number of layers of the base GNN")
    parser.add_argument("-gne", "--gnn-num-epochs", type=int, help="Number of epochs for pretraining")
    parser.add_argument("-ges", "--gnn-eval-step", type=int, help="Evaluation step for pretrained model")
    parser.add_argument("-ghd", "--gnn-h-dim", type=int, help="Hidden dim of the GNN")
    parser.add_argument("-gl", "--gnn-lr", type=float, help="Learning rate for pretraining the gnn")
    parser.add_argument("-gss", "--gnn-step-size", type=int, help="Learning rate step size for pretraining the gnn")
    parser.add_argument("-gg", "--gnn-gamma", type=float, help="Learning rate gamma for pretraining the gnn")
    parser.add_argument("-gbs", "--gnn-batch-size", type=int, help="Batch size for pretraining")
    parser.add_argument("-gn", "--gnn-dropout", type=float, help="Dropout for GNN")
    parser.add_argument("-pl", "--projector-lr", type=float, help="Learning rate for pretraining the projector")
    parser.add_argument("-pss", "--projector-step-size", type=int, help="Learning rate step size for pretraining the projector")
    parser.add_argument("-pwd", "--projector-weight-decay", type=float, help="Rate of regularization")
    parser.add_argument("-pg", "--projector-gamma", type=float, help="Learning rate gamma for pretraining the projector")
    parser.add_argument("-pne", "--projector-num-epochs", type=int, help="Number of epochs for pretraining the projector")
    parser.add_argument("-ct", "--contrastive-temperature", type=float, help="contrastive learning temperature")
    parser.add_argument("-pcrw", "--projector-contrastive-w", type=float, help="contrastive loss weights")
    parser.add_argument("-pcew", "--projector-context-w", type=float, help="context loss weights")
    parser.add_argument("-pmiw", "--projector-mutualinfo-w", type=float, help="mutual information loss weights")
    parser.add_argument("-pwb", "--proj-with-backward", action='store_true')
    parser.add_argument('-nv', '--not-verbose', action='store_true')
    parser.add_argument('-wno', '--write-new-output', action='store_true')
    parser.add_argument("-tt", "--total-iters", type=int, help="Total number of trials with random initialization of datasets")
    parser.add_argument("--seed", nargs='*', type=int, default=[], help="Seed for random")
    parser.add_argument("-cff", "--config-from-file", type=str, default="", help="Config file to read from")
    parser.add_argument('-ld', '--log-dir', default=None, type=str)
    parser.add_argument('-lbs', '--llm-batch-size', type=int)
    parser.add_argument('-mnen', '--max-num-eval-nodes', type=int)
    parser.add_argument('-mnfw', '--max-num-frequent-words', type=int)
    parser.add_argument('-nmh', '--neighborhood-max-hops', type=int)
    parser.add_argument('-lnbs', '--llm-node-batch-size', type=int)
    parser.add_argument('-sge', '--save-generation-every', type=int)
    parser.add_argument('-nt', '--num-tokens', type=int)
    parser.add_argument("-tts", "--train-test-split", nargs='*', type=float, help="[train_percentage, test_percentage]")
    parser.add_argument("-nm", "--normal-mode", type=str, help="Config file to save to")
    args = parser.parse_args()
    main(args)