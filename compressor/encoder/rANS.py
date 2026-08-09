#1.Run QualGRU's forward pass and save the probability distribution outputted at every single timestep in sequential order
#2.We then reverse the quality score sequence and pass it in to the encoder; this is done because the decoder decodes in reverse so if we encode backwards the decoder produces the sequence forward
	#-We need to set upper and lower limits for the state integer x
	#- M = 2^16, there are 50 symbols, and we need to set our initial x0 to the lower limit l
	#- At each timestep we take the probability distribution and turn it into a frequency table in which the sum of all frequencies adds up to M
	#- We use floors and largest remainders when converting the probabilities into frequencies, we also calculate the cumulative frequencies. We also enforce f>=1 for all s
	#- At each timestep, the frequency table and the encoding formula are used to push each symbol into the state.
		#- If the state is going to exceed the upper bound if the next symbol is pushed then we write to the output buffer
		#- The result of encoding is the final state integer x and the entirity of the output buffer

#3.For decoding we use the probability distributions outputted at every single timestep based on all previous symbols it just decoded, the final state integer x, and the entirity of the output buffer.
 	#-there is an agreed upon initial hidden state bw encoder and decoder that allows the decoder to figure out the "first" symbol in the sequence
	#-The probability distributions are converted to frequency tables just like before
	#-the decoding formula is used
		#-again there is a limit for the state integer x, it can't stay below the lower limit l
		#- if the lower limit is hit then we read from the output buffer
		# -this outputs the entirity of the original sequence "forward"


#The output buffer has to be written onto forward but read backwards

##########################################################################################
#encode
def encode():
	#run QualGRU forward pass on qscores+bases, will need to pass in file path via command line argument. Probability distributions at each timestep have to be saved
	#We walk through the q scores in reverse when encoding
	lower_limit = 2^23
	upper_limit = 2^31
	M = 2^16
	x = lower_limit
	for i in range(len(q_scores)):
		#convert QualGRU's prob table into a frequency table enforcing fs>=1, using floors, and largest remainders. Assert all frequencies add up to M, and this also gets cumfreq
		#while encoding next symbol is going to result in x > upper_limit: emit to output buffer whose file path will also be defined in the command line
		#use encode formula x = M*floor(x/fs) + (x%fs) + cs
	return 0

def decode():
	#use initial hidden state along with all other constants to decode the first symbol
	while x != lower_limit:
		#run QualGRU forward pass on all symbols to get probability table for next one -->convert to frequency table
		#while state integer is lower than lower_limit: read from output buffer BUT BACKWARDS
		#use decoding formula x = fs*floor(x/M) + (x % M -cs)
		#save symbols to file whose path will be specified in command line

	return 0
