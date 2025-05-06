# Trainer

- every trainer process continuously reads files with token sequences
- every trainer process reads shards from one topic, where topic = (actor, channel)
    - every channel is a folder, e.g. `text/1/15` is channel 15 by actor 1
- we checkpoint the training after every trainer has fully read a number of shards
- trainer process 0 keeps track of when it is time to broadcast the new weights

# Actor
- every actor loads entire dataset in memory
- the agent runner thread call the APIs, evaluates the tapes, and publishes data and stats in a queue
- the main thread reads the queue, write to the filesystem and to wandb
- another thread will listen on the LLM and Trainer status
- the main thread will stop putting tasks in the queue when it sees that LLM weight update is pending and max allowed lag is reached

# Preprocessor
- takes input from streams produced by the actor
- processes data: computes advantages, adds reference logprobs, etc
- outputs to the stream for the trainer
- logs statistics to wandb

# Orchestrator

Requirements
- no reward model
- 1 actor, all actor llms colocated at the same node with actor, localhost http requests are enough
- 1 preprocessor, same as assumptions as above ^^
- many nodes
- place the actor, place the preprocessor, use the rest for training
- resumable from a checkpoint but the old streams are deleted (archived)
- trainer uses streams to learn when inference servers are ready

Logs & configs
- orchestrator is a hydra script
- orchestrator saves its configuration to $EXP_DIR/conf/exp_config.yaml
- orchestrator launches all processeses with --config-dir $EXP_DIR --config-name exp_config 
- orchestrator uses whoami: section to tell the processes about their replica idx (rank) and rendezvous parameters
