---@diagnostic disable: need-check-nil, undefined-global
-- MarI/O by SethBling
-- Feel free to use this code, but please do not redistribute it.
-- Intended for use with the BizHawk emulator and Super Mario World or Super Mario Bros. ROM.
-- For SMW, make sure you have a save state named "DP1.state" at the beginning of a level,
-- and put a copy in both the Lua folder and the root directory of BizHawk.
ROMName1942 = "1942 (JU) [!]"
StateFile1942 = "1942.state"

if gameinfo.getromname() == ROMName1942 then
	Filename = StateFile1942
	ButtonNames = {
		"B",
		"Left",
		"Right",
		"Up",
		"Down"
	}
end

SCORE_START = 0x0427
SCORE_END = 0x042C
LEVEL = 0x0438
SHIP_ALIVE = 0x03B1

SPRITE_BUFFER_START = 0x0200
SHIP_ADDRESS_START = 0x0230
SHIP_ADDRESS_END = 0x0243
SPRITE_BUFFER_END = 0x02FF
PLANE_X_ADDRESS = 0x0237
PLANE_Y_ADDRESS = 0x0234
PLANE_X = 0
PLANE_Y = 0
BULLET_Y_BUFFER = 0x04AA
BULLET_X_BUFFER = 0x04C3
USELESS_SPRITES_SET = {}

BoxRadius = 1
POV_WIDTH = 255
POV_HEIGHT = 240
INPUT_SIZE = (POV_HEIGHT + 1) * (POV_WIDTH + 1)
INPUTS = INPUT_SIZE + 1
OUTPUTS = #ButtonNames
POPULATION = 300
DELTA_DISJOINT = 2.0
DELTA_WEIGHTS = 0.4
DELTA_THRESHOLD = 1.0
STALE_SPECIES = 15
MUTATE_CONNECTIONS_CHANCE = 0.25
PERTURB_CHANCE = 0.90
CROSSOVER_CHANCE = 0.75
LINK_MUTATION_CHANCE = 2.0
NODE_MUTATION_CHANCE = 0.50
BIAS_MUTATION_CHANCE = 0.40
STEP_SIZE = 0.1
DISABLE_MUTATION_CHANCE = 0.4
ENABLE_MUTATION_CHANCE = 0.2
TIMEOUT_CONSTANT = 300
MAXIMUM_NODES = 1000000

math.randomseed(os.time())

function GetScore()
	if gameinfo.getromname() == ROMName1942 then
		local score = 0
		local magnitude = 100000
		for i = SCORE_START, SCORE_END do
			score = score + (memory.readbyte(i) * magnitude)
			magnitude = magnitude / 10
		end
		return score
	end
end

function GetLevel()
	if gameinfo.getromname() == ROMName1942 then
		return memory.readbyte(LEVEL) + 1
	end
end

function GetFitness()
	if gameinfo.getromname() == ROMName1942 then
		return (Pool.currentFrame / 2 + GetScore()) * GetLevel()
	end
end

function GetPositions()
	if gameinfo.getromname() == ROMName1942 then
		PLANE_X = memory.readbyte(PLANE_X_ADDRESS)
		PLANE_Y = memory.readbyte(PLANE_Y_ADDRESS)
	end
end

function GetSprites()
	local sprites = {}
	for i = SPRITE_BUFFER_START, SPRITE_BUFFER_END, 4 do
		local sprite = memory.readbyte(i + 1)
		if USELESS_SPRITES_SET[sprite] ~= true then
			local spriteX = memory.readbyte(i + 3)
			local spriteY = memory.readbyte(i)
			if sprites[spriteX] == nil then sprites[spriteX] = {} end
			sprites[spriteX][spriteY] = true
		end
	end
	for i = 0, 7 do
		local bulletY = memory.readbyte(BULLET_Y_BUFFER + i)
		local bulletX = memory.readbyte(BULLET_X_BUFFER + i)
		if sprites[bulletX] == nil then sprites[bulletX] = {} end
		sprites[bulletX][bulletY] = true
	end
	return sprites
end

function GetInputs()
	GetPositions()

	local sprites = GetSprites()
	local inputs = {}

	for Y = 0, POV_HEIGHT do
		for X = 0, POV_WIDTH do
			inputs[#inputs + 1] = 0
			if sprites[X] ~= nil then
				if sprites[X][Y] == true then
					inputs[#inputs] = -1
				end
			end
			if X == PLANE_X and Y == PLANE_Y then
				inputs[#inputs] = 1
			end
		end
	end
	return inputs
end

function IsDead()
	return memory.readbyte(SHIP_ALIVE) == 1
end

--NO MODIFICATION NEEDED
function Sigmoid(x)
	return 2 / (1 + math.exp( -4.9 * x)) - 1
end

function NewInnovation()
	Pool.innovation = Pool.innovation + 1
	return Pool.innovation
end

function NewPool()
	local pool = {}
	pool.species = {}
	pool.generation = 0
	pool.innovation = OUTPUTS
	pool.currentSpecies = 1
	pool.currentGenome = 1
	pool.currentFrame = 0
	pool.maxFitness = 0

	return pool
end

function NewSpecies()
	local species = {}
	species.topFitness = 0
	species.staleness = 0
	species.genomes = {}
	species.averageFitness = 0

	return species
end

function NewGenome()
	local genome = {}
	genome.genes = {}
	genome.fitness = 0
	genome.adjustedFitness = 0
	genome.network = {}
	genome.maxneuron = 0
	genome.globalRank = 0
	genome.mutationRates = {}
	genome.mutationRates["connections"] = MUTATE_CONNECTIONS_CHANCE
	genome.mutationRates["link"] = LINK_MUTATION_CHANCE
	genome.mutationRates["bias"] = BIAS_MUTATION_CHANCE
	genome.mutationRates["node"] = NODE_MUTATION_CHANCE
	genome.mutationRates["enable"] = ENABLE_MUTATION_CHANCE
	genome.mutationRates["disable"] = DISABLE_MUTATION_CHANCE
	genome.mutationRates["step"] = STEP_SIZE

	return genome
end

function CopyGenome(genome)
	local genome2 = NewGenome()
	for g = 1, #genome.genes do
		table.insert(genome2.genes, CopyGene(genome.genes[g]))
	end
	genome2.maxneuron = genome.maxneuron
	genome2.mutationRates["connections"] = genome.mutationRates["connections"]
	genome2.mutationRates["link"] = genome.mutationRates["link"]
	genome2.mutationRates["bias"] = genome.mutationRates["bias"]
	genome2.mutationRates["node"] = genome.mutationRates["node"]
	genome2.mutationRates["enable"] = genome.mutationRates["enable"]
	genome2.mutationRates["disable"] = genome.mutationRates["disable"]

	return genome2
end

function BasicGenome()
	local genome = NewGenome()
	local innovation = 1

	genome.maxneuron = INPUTS
	Mutate(genome)

	return genome
end

function NewGene()
	local gene = {}
	gene.into = 0
	gene.out = 0
	gene.weight = 0.0
	gene.enabled = true
	gene.innovation = 0

	return gene
end

function CopyGene(gene)
	local gene2 = NewGene()
	gene2.into = gene.into
	gene2.out = gene.out
	gene2.weight = gene.weight
	gene2.enabled = gene.enabled
	gene2.innovation = gene.innovation

	return gene2
end

function NewNeuron()
	local neuron = {}
	neuron.incoming = {}
	neuron.value = 0.0

	return neuron
end

function GenerateNetwork(genome)
	local network = {}
	network.neurons = {}

	for i = 1, INPUTS do
		network.neurons[i] = NewNeuron()
	end

	for o = 1, OUTPUTS do
		network.neurons[MAXIMUM_NODES + o] = NewNeuron()
	end

	table.sort(genome.genes, function(a, b)
		return (a.out < b.out)
	end)
	for i = 1, #genome.genes do
		local gene = genome.genes[i]
		if gene.enabled then
			if network.neurons[gene.out] == nil then
				network.neurons[gene.out] = NewNeuron()
			end
			local neuron = network.neurons[gene.out]
			table.insert(neuron.incoming, gene)
			if network.neurons[gene.into] == nil then
				network.neurons[gene.into] = NewNeuron()
			end
		end
	end

	genome.network = network
end

function EvaluateNetwork(network, inputs)
	table.insert(inputs, 1)
	if #inputs ~= INPUTS then
		console.writeline("Incorrect number of neural network INPUTS.")
		return {}
	end

	for i = 1, INPUTS do
		network.neurons[i].value = inputs[i]
	end

	for _, neuron in pairs(network.neurons) do
		local sum = 0
		for j = 1, #neuron.incoming do
			local incoming = neuron.incoming[j]
			local other = network.neurons[incoming.into]
			sum = sum + incoming.weight * other.value
		end

		if #neuron.incoming > 0 then
			neuron.value = Sigmoid(sum)
		end
	end

	local outputs = {}
	for o = 1, OUTPUTS do
		local button = "P1 " .. ButtonNames[o]
		if network.neurons[MAXIMUM_NODES + o].value > 0 then
			outputs[button] = true
		else
			outputs[button] = false
		end
	end

	return outputs
end

function Crossover(g1, g2)
	-- Make sure g1 is the higher fitness genome
	if g2.fitness > g1.fitness then
		local tempg = g1
		g1 = g2
		g2 = tempg
	end

	local child = NewGenome()

	local innovations2 = {}
	for i = 1, #g2.genes do
		local gene = g2.genes[i]
		innovations2[gene.innovation] = gene
	end

	for i = 1, #g1.genes do
		local gene1 = g1.genes[i]
		local gene2 = innovations2[gene1.innovation]
		if gene2 ~= nil and math.random(2) == 1 and gene2.enabled then
			table.insert(child.genes, CopyGene(gene2))
		else
			table.insert(child.genes, CopyGene(gene1))
		end
	end

	child.maxneuron = math.max(g1.maxneuron, g2.maxneuron)

	for mutation, rate in pairs(g1.mutationRates) do
		child.mutationRates[mutation] = rate
	end

	return child
end

function RandomNeuron(genes, nonInput)
	local neurons = {}
	if not nonInput then
		for i = 1, INPUTS do
			neurons[i] = true
		end
	end
	for o = 1, OUTPUTS do
		neurons[MAXIMUM_NODES + o] = true
	end
	for i = 1, #genes do
		if (not nonInput) or genes[i].into > INPUTS then
			neurons[genes[i].into] = true
		end
		if (not nonInput) or genes[i].out > INPUTS then
			neurons[genes[i].out] = true
		end
	end

	local count = 0
	for _, _ in pairs(neurons) do
		count = count + 1
	end
	local n = math.random(1, count)

	for k, v in pairs(neurons) do
		n = n - 1
		if n == 0 then
			return k
		end
	end

	return 0
end

function ContainsLink(genes, link)
	for i = 1, #genes do
		local gene = genes[i]
		if gene.into == link.into and gene.out == link.out then
			return true
		end
	end
end

function PointMutate(genome)
	local step = genome.mutationRates["step"]

	for i = 1, #genome.genes do
		local gene = genome.genes[i]
		if math.random() < PERTURB_CHANCE then
			gene.weight = gene.weight + math.random() * step * 2 - step
		else
			gene.weight = math.random() * 4 - 2
		end
	end
end

function LinkMutate(genome, forceBias)
	local neuron1 = RandomNeuron(genome.genes, false)
	local neuron2 = RandomNeuron(genome.genes, true)

	local newLink = NewGene()
	if neuron1 <= INPUTS and neuron2 <= INPUTS then
		--Both input nodes
		return
	end
	if neuron2 <= INPUTS then
		-- Swap output and input
		local temp = neuron1
		neuron1 = neuron2
		neuron2 = temp
	end

	newLink.into = neuron1
	newLink.out = neuron2
	if forceBias then
		newLink.into = INPUTS
	end

	if ContainsLink(genome.genes, newLink) then
		return
	end
	newLink.innovation = NewInnovation()
	newLink.weight = math.random() * 4 - 2

	table.insert(genome.genes, newLink)
end

function NodeMutate(genome)
	if #genome.genes == 0 then
		return
	end

	genome.maxneuron = genome.maxneuron + 1

	local gene = genome.genes[math.random(1, #genome.genes)]
	if not gene.enabled then
		return
	end
	gene.enabled = false

	local gene1 = CopyGene(gene)
	gene1.out = genome.maxneuron
	gene1.weight = 1.0
	gene1.innovation = NewInnovation()
	gene1.enabled = true
	table.insert(genome.genes, gene1)

	local gene2 = CopyGene(gene)
	gene2.into = genome.maxneuron
	gene2.innovation = NewInnovation()
	gene2.enabled = true
	table.insert(genome.genes, gene2)
end

function EnableDisableMutate(genome, enable)
	local candidates = {}
	for _, gene in pairs(genome.genes) do
		if gene.enabled == not enable then
			table.insert(candidates, gene)
		end
	end

	if #candidates == 0 then
		return
	end

	local gene = candidates[math.random(1, #candidates)]
	gene.enabled = not gene.enabled
end

function Mutate(genome)
	for mutation, rate in pairs(genome.mutationRates) do
		if math.random(1, 2) == 1 then
			genome.mutationRates[mutation] = 0.95 * rate
		else
			genome.mutationRates[mutation] = 1.05263 * rate
		end
	end

	if math.random() < genome.mutationRates["connections"] then
		PointMutate(genome)
	end

	local p = genome.mutationRates["link"]
	while p > 0 do
		if math.random() < p then
			LinkMutate(genome, false)
		end
		p = p - 1
	end

	p = genome.mutationRates["bias"]
	while p > 0 do
		if math.random() < p then
			LinkMutate(genome, true)
		end
		p = p - 1
	end

	p = genome.mutationRates["node"]
	while p > 0 do
		if math.random() < p then
			NodeMutate(genome)
		end
		p = p - 1
	end

	p = genome.mutationRates["enable"]
	while p > 0 do
		if math.random() < p then
			EnableDisableMutate(genome, true)
		end
		p = p - 1
	end

	p = genome.mutationRates["disable"]
	while p > 0 do
		if math.random() < p then
			EnableDisableMutate(genome, false)
		end
		p = p - 1
	end
end

function Disjoint(genes1, genes2)
	local i1 = {}
	for i = 1, #genes1 do
		local gene = genes1[i]
		i1[gene.innovation] = true
	end

	local i2 = {}
	for i = 1, #genes2 do
		local gene = genes2[i]
		i2[gene.innovation] = true
	end

	local disjointGenes = 0
	for i = 1, #genes1 do
		local gene = genes1[i]
		if not i2[gene.innovation] then
			disjointGenes = disjointGenes + 1
		end
	end

	for i = 1, #genes2 do
		local gene = genes2[i]
		if not i1[gene.innovation] then
			disjointGenes = disjointGenes + 1
		end
	end

	local n = math.max(#genes1, #genes2)

	return disjointGenes / n
end

function Weights(genes1, genes2)
	local i2 = {}
	for i = 1, #genes2 do
		local gene = genes2[i]
		i2[gene.innovation] = gene
	end

	local sum = 0
	local coincident = 0
	for i = 1, #genes1 do
		local gene = genes1[i]
		if i2[gene.innovation] ~= nil then
			local gene2 = i2[gene.innovation]
			sum = sum + math.abs(gene.weight - gene2.weight)
			coincident = coincident + 1
		end
	end

	return sum / coincident
end

function SameSpecies(genome1, genome2)
	local dd = DELTA_DISJOINT * Disjoint(genome1.genes, genome2.genes)
	local dw = DELTA_WEIGHTS * Weights(genome1.genes, genome2.genes)
	return dd + dw < DELTA_THRESHOLD
end

function RankGlobaly()
	local global = {}
	for s = 1, #Pool.species do
		local species = Pool.species[s]
		for g = 1, #species.genomes do
			table.insert(global, species.genomes[g])
		end
	end
	table.sort(global, function(a, b)
		return (a.fitness < b.fitness)
	end)

	for g = 1, #global do
		global[g].globalRank = g
	end
end

function CalculateAverageFitness(species)
	local total = 0

	for g = 1, #species.genomes do
		local genome = species.genomes[g]
		total = total + genome.globalRank
	end

	species.averageFitness = total / #species.genomes
end

function TotalAverageFitness()
	local total = 0
	for s = 1, #Pool.species do
		local species = Pool.species[s]
		total = total + species.averageFitness
	end

	return total
end

function CullSpecies(cutToOne)
	for s = 1, #Pool.species do
		local species = Pool.species[s]

		table.sort(species.genomes, function(a, b)
			return (a.fitness > b.fitness)
		end)

		local remaining = math.ceil(#species.genomes / 2)
		if cutToOne then
			remaining = 5
		end
		while #species.genomes > remaining do
			table.remove(species.genomes)
		end
	end
end

function BreedChild(species)
	local child = {}
	if math.random() < CROSSOVER_CHANCE then
		local g1 = species.genomes[math.random(1, #species.genomes)]
		local g2 = species.genomes[math.random(1, #species.genomes)]
		child = Crossover(g1, g2)
	else
		local g = species.genomes[math.random(1, #species.genomes)]
		child = CopyGenome(g)
	end

	Mutate(child)

	return child
end

function RemoveStaleSpecies()
	local survived = {}

	for s = 1, #Pool.species do
		local species = Pool.species[s]

		table.sort(species.genomes, function(a, b)
			return (a.fitness > b.fitness)
		end)

		if species.genomes[1].fitness > species.topFitness then
			species.topFitness = species.genomes[1].fitness
			species.staleness = 0
		else
			species.staleness = species.staleness + 1
		end
		if species.staleness < STALE_SPECIES or species.topFitness >= Pool.maxFitness then
			table.insert(survived, species)
		end
	end

	Pool.species = survived
end

function RemoveWeakSpecies()
	local survived = {}

	local sum = TotalAverageFitness()
	for s = 1, #Pool.species do
		local species = Pool.species[s]
		local breed = math.floor(species.averageFitness / sum * POPULATION)
		if breed >= 1 then
			table.insert(survived, species)
		end
	end

	Pool.species = survived
end

function AddToSpecies(child)
	local foundSpecies = false
	for s = 1, #Pool.species do
		local species = Pool.species[s]
		if not foundSpecies and SameSpecies(child, species.genomes[1]) then
			table.insert(species.genomes, child)
			foundSpecies = true
		end
	end

	if not foundSpecies then
		local childSpecies = NewSpecies()
		table.insert(childSpecies.genomes, child)
		table.insert(Pool.species, childSpecies)
	end
end

function NewGeneration()
	CullSpecies(false) -- Cull the bottom half of each species
	RankGlobaly()
	RemoveStaleSpecies()
	RankGlobaly()
	for s = 1, #Pool.species do
		local species = Pool.species[s]
		CalculateAverageFitness(species)
	end
	RemoveWeakSpecies()
	local sum = TotalAverageFitness()
	local children = {}
	for s = 1, #Pool.species do
		local species = Pool.species[s]
		local breed = math.floor(species.averageFitness / sum * POPULATION) - 1
		for i = 1, breed do
			table.insert(children, BreedChild(species))
		end
	end
	CullSpecies(true) -- Cull all but the top member of each species
	while #children + #Pool.species < POPULATION do
		local species = Pool.species[math.random(1, #Pool.species)]
		table.insert(children, BreedChild(species))
	end
	for c = 1, #children do
		local child = children[c]
		AddToSpecies(child)
	end

	Pool.generation = Pool.generation + 1

	WriteFile("backup." .. Pool.generation .. "." .. forms.gettext(SaveLoadFile))
end

function InitializePool()
	Pool = NewPool()

	for i = 1, POPULATION do
		local basic = BasicGenome()
		AddToSpecies(basic)
	end

	InitializeRun()
end

function ClearJoypad()
	Controller = {}
	for b = 1, #ButtonNames do
		Controller["P1 " .. ButtonNames[b]] = false
	end
	joypad.set(Controller)
end

function InitializeRun()
	savestate.load(Filename);
	Rightmost = {}
	Rightmost[0] = 18
	Rightmost[1] = 18
	Pool.currentFrame = 0
	Timeout = TIMEOUT_CONSTANT
	ClearJoypad()

	local species = Pool.species[Pool.currentSpecies]
	local genome = species.genomes[Pool.currentGenome]
	GenerateNetwork(genome)
	EvaluateCurrent()
end

function EvaluateCurrent()
	local species = Pool.species[Pool.currentSpecies]
	local genome = species.genomes[Pool.currentGenome]

	local inputs = GetInputs()
	Controller = EvaluateNetwork(genome.network, inputs)

	if Controller["P1 Left"] and Controller["P1 Right"] then
		Controller["P1 Left"] = false
		Controller["P1 Right"] = false
	end
	if Controller["P1 Up"] and Controller["P1 Down"] then
		Controller["P1 Up"] = false
		Controller["P1 Down"] = false
	end

	joypad.set(Controller)
end

if Pool == nil then
	InitializePool()
end


function NextGenome()
	Pool.currentGenome = Pool.currentGenome + 1
	if Pool.currentGenome > #Pool.species[Pool.currentSpecies].genomes then
		Pool.currentGenome = 1
		Pool.currentSpecies = Pool.currentSpecies + 1
		if Pool.currentSpecies > #Pool.species then
			NewGeneration()
			Pool.currentSpecies = 1
		end
	end
end

function FitnessAlreadyMeasured()
	local species = Pool.species[Pool.currentSpecies]
	local genome = species.genomes[Pool.currentGenome]

	return genome.fitness ~= 0
end

-- END NO MODIFICATION NEEDED

function DisplayGenome(genome)
	local network = genome.network
	local cells = {}
	local i = 1
	local cell = {}
	for dy = 0, POV_HEIGHT do
		for dx = 0, POV_WIDTH do
			cell = {}
			cell.x = dx
			cell.y = dy
			cell.value = network.neurons[i].value
			cells[i] = cell
			i = i + 1
		end
	end
	local biasCell = {}
	biasCell.x = 80
	biasCell.y = 110
	biasCell.value = network.neurons[INPUTS].value
	cells[INPUTS] = biasCell

	for o = 1, OUTPUTS do
		cell = {}
		cell.x = 220
		cell.y = 30 + 8 * o
		cell.value = network.neurons[MAXIMUM_NODES + o].value
		cells[MAXIMUM_NODES + o] = cell
		local color
		if cell.value > 0 then
			color = 0xFF0000FF
		else
			color = 0xFF000000
		end
		gui.drawText(223, 24 + 8 * o, ButtonNames[o], color, 9)
	end

	for n, neuron in pairs(network.neurons) do
		cell = {}
		if n > INPUTS and n <= MAXIMUM_NODES then
			cell.x = 140
			cell.y = 40
			cell.value = neuron.value
			cells[n] = cell
		end
	end

	for n = 1, 4 do
		for _, gene in pairs(genome.genes) do
			if gene.enabled then
				local c1 = cells[gene.into]
				local c2 = cells[gene.out]
				if gene.into > INPUTS and gene.into <= MAXIMUM_NODES then
					c1.x = 0.75 * c1.x + 0.25 * c2.x
					if c1.x >= c2.x then
						c1.x = c1.x - 40
					end
					if c1.x < 90 then
						c1.x = 90
					end

					if c1.x > 220 then
						c1.x = 220
					end
					c1.y = 0.75 * c1.y + 0.25 * c2.y
				end
				if gene.out > INPUTS and gene.out <= MAXIMUM_NODES then
					c2.x = 0.25 * c1.x + 0.75 * c2.x
					if c1.x >= c2.x then
						c2.x = c2.x + 40
					end
					if c2.x < 90 then
						c2.x = 90
					end
					if c2.x > 220 then
						c2.x = 220
					end
					c2.y = 0.25 * c1.y + 0.75 * c2.y
				end
			end
		end
	end

	gui.drawBox(0, 0, 0 + 255, 0 + 240,
		0xFF000000, 0x80808080)
	for n, cell in pairs(cells) do
		if n > INPUTS or cell.value ~= 0 then
			if cell.vale ~= 0 then
				local color = 0
				if cell.value == 1 then color = 0xFFFFFFFF end
				if cell.value == -1 then color = 0xFF000000 end
				gui.drawPixel(cell.x, cell.y, color)
			end
		end
	end
	for _, gene in pairs(genome.genes) do
		if gene.enabled then
			local c1 = cells[gene.into]
			local c2 = cells[gene.out]
			local opacity = 0xA0000000
			if c1.value == 0 then
				opacity = 0x20000000
			end

			local color = 0x80 - math.floor(math.abs(Sigmoid(gene.weight)) * 0x80)
			if gene.weight > 0 then
				color = opacity + 0x8000 + 0x10000 * color
			else
				color = opacity + 0x800000 + 0x100 * color
			end
			gui.drawLine(c1.x + 1, c1.y, c2.x - 3, c2.y, color)
		end
	end

	--gui.drawBox(49, 71, 51, 78, 0x00000000, 0x80FF0000)

	if forms.ischecked(ShowMutationRates) then
		local pos = 100
		for mutation, rate in pairs(genome.mutationRates) do
			gui.drawText(100, pos, mutation .. ": " .. rate, 0xFF000000, 10)
			pos = pos + 8
		end
	end
end

function WriteFile(filename)
	local file = io.open(filename, "w")
	file:write(Pool.generation .. "\n")
	file:write(Pool.maxFitness .. "\n")
	file:write(#Pool.species .. "\n")
	for n, species in pairs(Pool.species) do
		file:write(species.topFitness .. "\n")
		file:write(species.staleness .. "\n")
		file:write(#species.genomes .. "\n")
		for m, genome in pairs(species.genomes) do
			file:write(genome.fitness .. "\n")
			file:write(genome.maxneuron .. "\n")
			for mutation, rate in pairs(genome.mutationRates) do
				file:write(mutation .. "\n")
				file:write(rate .. "\n")
			end
			file:write("done\n")

			file:write(#genome.genes .. "\n")
			for l, gene in pairs(genome.genes) do
				file:write(gene.into .. " ")
				file:write(gene.out .. " ")
				file:write(gene.weight .. " ")
				file:write(gene.innovation .. " ")
				if (gene.enabled) then
					file:write("1\n")
				else
					file:write("0\n")
				end
			end
		end
	end
	file:close()
end

function SavePool()
	local filename = forms.gettext(SaveLoadFile)
	WriteFile(filename)
end

function LoadFile(filename)
	local file = io.open(filename, "r")
	Pool = NewPool()
	Pool.generation = file:read("*number")
	Pool.maxFitness = file:read("*number")
	forms.settext(MaxFitnessLabel, "Max Fitness: " .. math.floor(Pool.maxFitness))
	local numSpecies = file:read("*number")
	for s = 1, numSpecies do
		local species = NewSpecies()
		table.insert(Pool.species, species)
		species.topFitness = file:read("*number")
		species.staleness = file:read("*number")
		local numGenomes = file:read("*number")
		for g = 1, numGenomes do
			local genome = NewGenome()
			table.insert(species.genomes, genome)
			genome.fitness = file:read("*number")
			genome.maxneuron = file:read("*number")
			local line = file:read("*line")
			while line ~= "done" do
				genome.mutationRates[line] = file:read("*number")
				line = file:read("*line")
			end
			local numGenes = file:read("*number")
			for n = 1, numGenes do
				local gene = NewGene()
				table.insert(genome.genes, gene)
				local enabled
				gene.into, gene.out, gene.weight, gene.innovation, enabled = file:read("*number", "*number", "*number",
					"*number",
					"*number")
				if enabled == 0 then
					gene.enabled = false
				else
					gene.enabled = true
				end
			end
		end
	end
	file:close()

	while FitnessAlreadyMeasured() do
		NextGenome()
	end
	InitializeRun()
	Pool.currentFrame = Pool.currentFrame + 1
end

function LoadPool()
	local filename = forms.gettext(SaveLoadFile)
	LoadFile(filename)
end

function PlayTop()
	local maxfitness = 0
	local maxs, maxg
	for s, species in pairs(Pool.species) do
		for g, genome in pairs(species.genomes) do
			if genome.fitness > maxfitness then
				maxfitness = genome.fitness
				maxs = s
				maxg = g
			end
		end
	end

	Pool.currentSpecies = maxs
	Pool.currentGenome = maxg
	Pool.maxFitness = maxfitness
	forms.settext(MaxFitnessLabel, "Max Fitness: " .. math.floor(Pool.maxFitness))
	InitializeRun()
	Pool.currentFrame = Pool.currentFrame + 1
end

function OnExit()
	forms.destroy(Form)
end

function ProcessUselessSprites()
	USELESS_SPRITES = { 0x30, 0x31, 0x32, 0xFE, 0xFF }
	POWER_UPS = { 0x2D, 0x6F }

	for i = 0x00, 0x27 do
		USELESS_SPRITES_SET[i] = true
	end

	for i = 0xA3, 0xAE do
		USELESS_SPRITES_SET[i] = true
	end

	for i = 0xD0, 0xED do
		USELESS_SPRITES_SET[i] = true
	end

	for i = 0x33, 0x36 do
		USELESS_SPRITES_SET[i] = true
	end

	for i = 1, #USELESS_SPRITES do
		USELESS_SPRITES_SET[USELESS_SPRITES[i]] = true
	end
end

ProcessUselessSprites()

WriteFile("temp.Pool")

event.onexit(OnExit)

Form = forms.newform(200, 260, "Fitness")
MaxFitnessLabel = forms.label(Form, "Max Fitness: " .. math.floor(Pool.maxFitness), 5, 8)
ShowNetwork = forms.checkbox(Form, "Show Map", 5, 30)
ShowMutationRates = forms.checkbox(Form, "Show M-Rates", 5, 52)
RestartButton = forms.button(Form, "Restart", InitializePool, 5, 77)
SaveButton = forms.button(Form, "Save", SavePool, 5, 102)
LoadButton = forms.button(Form, "Load", LoadPool, 80, 102)
SaveLoadFile = forms.textbox(Form, Filename .. ".Pool", 170, 25, nil, 5, 148)
SaveLoadLabel = forms.label(Form, "Save/Load:", 5, 129)
PlayTopButton = forms.button(Form, "Play Top", PlayTop, 5, 170)
HideBanner = forms.checkbox(Form, "Hide Banner", 5, 190)

while true do
	local backgroundColor = 0xD0FFFFFF
	if not forms.ischecked(HideBanner) then
		gui.drawBox(0, 0, 300, 26, backgroundColor, backgroundColor)
	end

	local species = Pool.species[Pool.currentSpecies]
	local genome = species.genomes[Pool.currentGenome]

	if forms.ischecked(ShowNetwork) then
		DisplayGenome(genome)
	end

	if Pool.currentFrame % 5 == 0 then
		EvaluateCurrent()
	end

	joypad.set(Controller)

	--GetPositions()
	if PLANE_X ~= Rightmost[0] or PLANE_Y ~= Rightmost[1] then
		Rightmost[0] = PLANE_X
		Rightmost[1] = PLANE_Y
		Timeout = TIMEOUT_CONSTANT
	end

	Timeout = Timeout - 1

	local TimeoutBonus = Pool.currentFrame / 4
	local hasTimedOut = Timeout + TimeoutBonus <= 0
	local IsDead = IsDead()
	if IsDead or hasTimedOut then
		local fitness = GetFitness()
		genome.fitness = fitness

		if fitness > Pool.maxFitness then
			Pool.maxFitness = fitness
			forms.settext(MaxFitnessLabel, "Max Fitness: " .. math.floor(Pool.maxFitness))
			WriteFile("backup." .. Pool.generation .. "." .. forms.gettext(SaveLoadFile))
		end

		console.writeline("Gen " ..
		Pool.generation ..
		" species " .. Pool.currentSpecies .. " genome " .. Pool.currentGenome .. " fitness: " .. fitness)
		Pool.currentSpecies = 1
		Pool.currentGenome = 1
		while FitnessAlreadyMeasured() do
			NextGenome()
		end
		InitializeRun()
	end

	local measured = 0
	local total = 0
	for _, species in pairs(Pool.species) do
		for _, genome in pairs(species.genomes) do
			total = total + 1
			if genome.fitness ~= 0 then
				measured = measured + 1
			end
		end
	end
	if not forms.ischecked(HideBanner) then
		gui.drawText(0, 0,
			"Gen " ..
			Pool.generation ..
			" species " ..
			Pool.currentSpecies .. " genome " .. Pool.currentGenome .. " (" .. math.floor(measured / total * 100) .. "%)",
			0xFF000000, 11)
		gui.drawText(0, 12, "Fitness: " .. GetFitness(),
			0xFF000000, 11)
		gui.drawText(100, 12, "Max Fitness: " .. math.floor(Pool.maxFitness), 0xFF000000, 11)
	end

	Pool.currentFrame = Pool.currentFrame + 1

	emu.frameadvance();
end
