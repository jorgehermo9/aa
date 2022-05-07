
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean
using Random
using Statistics

using Plots
using DelimitedFiles
using Printf

include("src/scikit/p6.jl")


Random.seed!(100)

db_path = "dataset/db.jld2"
all_signals   = load(db_path, "all_signals");
all_labels = load(db_path, "all_labels");

all_signals = normalizeZeroMean(all_signals);

inputs = all_signals;
targets = all_labels;

folds = 10;

# Reuso codigo de las aproximaciones anteriores, creo el array con un solo modelo... 
models = [:CONV]

models_parameters = Vector{Vector{Dict{Any,Any}}}(undef, length(models));

acc_models = Vector{Vector{Float64}}(undef,length(models));
acc_std_models = Vector{Vector{Float64}}(undef,length(models));

recall_models = Vector{Vector{Float64}}(undef,length(models));
recall_std_models = Vector{Vector{Float64}}(undef,length(models));

specifity_models = Vector{Vector{Float64}}(undef,length(models));
specifity_std_models = Vector{Vector{Float64}}(undef,length(models));

f1_models = Vector{Vector{Float64}}(undef,length(models));
f1_std_models = Vector{Vector{Float64}}(undef,length(models));

#RNA

funcionTransferenciaCapasConvolucionales = relu;
topologies =[
	Chain(
		Conv((2,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		Conv((2,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		Conv((2,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((2,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		Conv((2,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		Conv((2,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((2,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		Conv((2,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((2,)),
		Conv((2,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((2,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((2,)),
		Conv((2,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		Conv((2,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((3,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		Conv((3,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		Conv((3,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((5,), 1=>4, pad=2, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		Conv((5,), 4=>8, pad=2, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		Conv((5,), 8=>16, pad=2, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((2,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		Conv((2,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		Conv((2,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((3,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		Conv((3,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		Conv((3,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((3,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((2,)),
		Conv((3,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		Conv((3,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
	Chain(
		Conv((3,), 1=>4, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((5,)),
		Conv((3,), 4=>8, pad=1, funcionTransferenciaCapasConvolucionales),
		MeanPool((2,)),
		Conv((3,), 8=>16, pad=1, funcionTransferenciaCapasConvolucionales),
		MaxPool((5,)),
		x -> reshape(x, :, size(x, 3)),
		Dense(160, length(unique(all_labels))),
		softmax
	),
]
conv_parameters = Vector{Dict{Any,Any}}(undef, length(topologies));

for i in 1:length(topologies)
	topology_parameters = Dict();
	topology_parameters["ann"] = topologies[i];
	topology_parameters["conv_executions"] = 1;
	conv_parameters[i] = topology_parameters;
end
models_parameters[1] = conv_parameters

# Models cross validation

for i in 1:length(models)
	configurations = models_parameters[i]
	num_configurations = length(configurations)

	acc_configurations = zeros(num_configurations)
	acc_std_configurations = zeros(num_configurations)

	recall_configurations = zeros(num_configurations)
	recall_std_configurations = zeros(num_configurations)

	specifity_configurations = zeros(num_configurations)
	specifity_std_configurations = zeros(num_configurations)

	f1_configurations = zeros(num_configurations)
	f1_std_configurations = zeros(num_configurations)

	for j in 1:num_configurations
		println("Training $(models[i]) (config $(j))")
		configuration_parameters = configurations[j]
		((acc_configurations[j],acc_std_configurations[j])), (recall_configurations[j],recall_std_configurations[j]),
			(specifity_configurations[j],specifity_std_configurations[j]), (f1_configurations[j],f1_std_configurations[j]) = 
				modelCrossValidation(models[i],configuration_parameters,inputs,targets,kfolds=folds)
	end
	acc_models[i] = acc_configurations
	acc_std_models[i] = acc_std_configurations

	recall_models[i] = recall_configurations
	recall_std_models[i] = recall_std_configurations

	specifity_models[i] = specifity_configurations
	specifity_std_models[i] = specifity_std_configurations

	f1_models[i] = f1_configurations
	f1_std_models[i] = f1_std_configurations
end

# Resultados 
# for i in 1:length(models)

# 	println("--------------------------------")
# 	println("Model: $(models[i])")
# 	configurations = models_parameters[i]
# 	for j in 1:length(configurations)
# 		println("")
# 		configuration_parameters = configurations[j]
# 		for key in keys(configuration_parameters)
# 			println("$(key): $(configuration_parameters[key])")
# 		end
# 		println("Accuracy: $(round(acc_models[i][j],digits=5)) ± $(round(acc_std_models[i][j],digits=5))")
# 		println("F1-Score: $(round(f1_models[i][j],digits=5)) ± $(round(f1_std_models[i][j],digits=5))")
# 		println("")
# 	end
# 	println("--------------------------------")

# end

# Resultados para latex
for i in 1:length(models)

	if models[i] == :CONV
		config_name ="Arquitectura"
		caption="CONV"
	else
		config_name=""
	end

	println("----------------------------")
	println("$(models[i])\n")
	println("\\begin{table}[!ht]")
	println("\\caption{Resultados $(caption)}")
	println("\\centering")
	println("\t \\begin{tabular}{||c c c||} ")
	println("\t\t \\hline ")
	println("\t\t $(config_name) & F1-Score & Precisión  \\\\ [0.5ex]  ")
	println("\t\t \\hline\\hline")
	configurations = models_parameters[i]
	for j in 1:length(configurations)
		configuration_parameters = configurations[j]
		if models[i] == :CONV
			config = "conv\\textsubscript{$(j)}"
		else
			config=""
		end
		println("\t\t $(config) & \$$(round(f1_models[i][j],digits=5)) \\pm $(round(f1_std_models[i][j],digits=5))\$ & \$$(round(acc_models[i][j],digits=5)) \\pm $(round(acc_std_models[i][j],digits=5))\$ \\\\")
		println("\t\t \\hline ")
	end
	println("\t \\end{tabular}")
	println("\\label{Tab:$(models[i])} ")
	println("\\end{table} ")

	println("\n----------------------------")
end

best_config=(1,1);
best_mean_f1 = f1_models[1][1];
best_std_f1 = f1_std_models[1][1];
for i in 1:length(models)
	model_configurations = models_parameters[i]
	for j in 1:length(model_configurations)
		if f1_models[i][j] == best_mean_f1
			# A igual f1, elijo la que tenga menos std
			if f1_std_models[i][j] < best_std_f1
				global best_std_f1 = f1_std_models[i][j]
				global best_config = (i,j)
			end
		elseif f1_models[i][j] > best_mean_f1
			global best_mean_f1 = f1_models[i][j]
			global best_config = (i,j)
		end
	end
end
(best_model,best_model_config) = best_config
println("Best configuration\n")
println("Model: $(models[best_model])")
model_configuration = models_parameters[best_model][best_model_config]
println("Hyperparameters:")
for key in keys(model_configuration)
	println("\t$(key): $(model_configuration[key])")
end
println("Accuracy: $(round(acc_models[best_model][best_model_config],digits=5)) ± $(round(acc_std_models[best_model][best_model_config],digits=5))")
println("F1-Score: $(round(f1_models[best_model][best_model_config],digits=5)) ± $(round(f1_std_models[best_model][best_model_config],digits=5))")


# Resultados para el mejor modelo

# Un 20% de las instancias son para generar la matriz de confusión (test)

classes = unique(targets)

inputs = reshape(inputs,(size(inputs,1),1,size(inputs,2)))
targets = oneHotEncoding(targets)'

test_ratio = 0.2
train_idx,test_idx = holdOut(size(targets,2),test_ratio)


train_inputs,train_targets = inputs[:,:,train_idx],targets[:,train_idx]
test_inputs,test_targets = inputs[:,:,test_idx],targets[:,test_idx]


parameters = models_parameters[best_model][best_model_config]

if models[best_model] == :CONV	
	
	train = (train_inputs,train_targets);
	test = (test_inputs, test_targets);
	
	(ann, train_vector, test_vector) = trainConv(parameters, train,test);

	test_outputs = ann(test_inputs)

	plotlyjs();
	g = plot();
	plot!(g,0:(length(train_vector)-1),train_vector,xaxis="Epoch",yaxis="Loss",color=:red,label="train");
	plot!(g,0:(length(test_vector)-1),test_vector,xaxis="Epoch",yaxis="Loss",color=:green,label="test");

	# display(g);
	path = "best_conv_train.svg"
	savefig(g,path)
	println("ANN training plot saved to $(path)")
end

println("------------------------------")
println("Results using whole dataset for best model with $(test_ratio) test ratio:\n")

(acc,_,recall,specifity,_,
		_,f1,matrix) =confusionMatrix(onecold(test_outputs),onecold(test_targets),macro_strat)


# Print confusion matrix
@printf "%-3s " ""
for i in 1:length(classes)
	@printf "%-3s " classes[i]
end
@printf "\n"
for i in 1:size(matrix,1)
	@printf "%-3s " classes[i]
	for j in 1:size(matrix,2)
		@printf "%-3d " matrix[i,j]
	end
	@printf "\n"
end

println("Accuracy: $(round(acc,digits=5))")
println("F1-Score: $(round(f1,digits=5))")
println("Recall: $(round(recall,digits=5))")
println("Specifity: $(round(specifity,digits=5))")
