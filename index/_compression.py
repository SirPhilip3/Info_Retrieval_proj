from bitarray import bitarray
from bitarray.util import ba2int, int2ba

class IndexCompressor:

    def __init__(self, index_obj=None):
        self.index_obj = index_obj
        self.pfor_delta_k = 8  # Default value for k in PForDelta encoding
        self.pfor_delta_block_size = 128  # Default block size for PForDelta encoding
        self.pfor_delta_extra_strategy = "None"  # Default strategy for extending the last block in PForDelta encoding

    def clear_compressor(self):
        self.pfor_delta_k = 8
        self.pfor_delta_block_size = 128
        self.pfor_delta_extra_strategy = "None"

    def apply_index_transformation(self, transform_func, is_decompressing=False):
        
        if self.index_obj.compression_method == "pfor_delta":
            print(f"Handling PForDelta with is_decompressing={is_decompressing}")
            self.handle_pfor_delta(is_decompressing=is_decompressing)
            return  # Skip the rest of the loop for PForDelta since it's handled separately
        
        for term, posting_list in self.index_obj.index.items():

            # we need to do this since pfor_delta_encode_bit and pfor_delta_decode_bit are not standard functions and require special handling
            
            compressed_postings = []

            def run_transform(item):
                output = transform_func(item)
                # If it's a generator decoder, extract the yielded value
                return next(output) if is_decompressing else output

            if self.index_obj.removed_tf: #! can be optimzed 
                for doc_id in posting_list:
                    compressed_doc_id = run_transform(doc_id)
                    compressed_postings.append(compressed_doc_id)
            else:
                for doc_id, tf in posting_list:
                    # Gamma encode doc_id and tf
                    compressed_doc_id = run_transform(doc_id)
                    compressed_tf = run_transform(tf)
                    compressed_postings.append((compressed_doc_id, compressed_tf))
            self.index_obj.index[term] = compressed_postings

    def transform_unary(self, number):
        return bitarray([1] * number + [0])

    def gamma_encode(self, number):

        if number <= 0:
            raise ValueError("Gamma encoding requires positive integers.")
            
        length = number.bit_length()  # Fast O(1) way to find binary length
        
        # 1. Unary prefix for (length - 1)
        res = self.transform_unary(length - 1)
        
        # 2. Append the binary remainder (all bits after the leading 1)
        if length > 1:
            res.extend(int2ba(number)[1:])

        # print(f"Gamma encoding number: {number}, Length: {length}, Unary: {res[:length-1]}, Remainder: {res[length-1:]}")
            
        return res

    def delta_encode(self, number):
        if number <= 0:
            raise ValueError("Delta encoding requires positive integers.")
            
        length = number.bit_length()
        
        # 1. Gamma encode the length
        res = self.gamma_encode(length)
        
        # 2. Append the binary remainder
        if length > 1:
            res.extend(int2ba(number)[1:])
            
        return res

    def varbyte_encode(self, number):
        if number == 0:
            return bitarray([1, 0, 0, 0, 0, 0, 0, 0])

        chunks = []
        is_first = True
        
        # Extract 7-bit blocks from right to left using fast bitwise shifts
        while number > 0:
            chunk = number & 0x7F  # Get lower 7 bits
            
            # Construct the 8-bit block directly: [Flag Bit] + [7 Payload Bits]
            block = bitarray([1 if is_first else 0])
            block.extend(int2ba(chunk, length=7))
            
            chunks.append(block)
            number >>= 7  # Shift right by 7 bits
            is_first = False

        # Since we went right-to-left (LSB to MSB), reverse chunks into a final bitarray
        res = bitarray()
        for block in reversed(chunks):
            res.extend(block)
            
        return res

    def gamma_decode(self, encoded_bytes, verbose=False): #! use yield

        # ptr = 0
        total_length = len(encoded_bytes)

        while len(encoded_bytes) > 0:
        
            remaining_length = len(encoded_bytes)
            if verbose and remaining_length % 1000 == 0:
                percentage = (remaining_length / total_length) * 100
                print(f"\rProcessing: {percentage:6.1f}% [{remaining_length}/{total_length}]", end="", flush=True)

            try :
                to_read = encoded_bytes.index(0)  # Find the first 0 (end of unary prefix)
            except ValueError:
                # Break out if no 0 is found to prevent an infinite loop on corrupt data
                raise ValueError("Invalid gamma encoding: no terminating 0 found in unary prefix.")
            
            remainder_start = to_read + 1
            remainder_end = remainder_start + to_read

            res = bitarray([1])
            res.extend(encoded_bytes[remainder_start:remainder_end])  # Add the remainder bits
            
            # ptr = remainder_end  # Move the pointer past the remainder bits
            del encoded_bytes[:remainder_end]  # Remove the processed bits from the front of the list

            yield ba2int(res)

    def delta_decode(self, encoded_bytes, verbose=False): #! use yield
        #! since we are use yield we cant call gamma decode directly , think about this for reuse of code
        # ptr = 0
        total_length = len(encoded_bytes)
        gamme_gen = self.gamma_decode(encoded_bytes, verbose=verbose)  # Create a generator for gamma decoding

        while len(encoded_bytes) > 0:
            
            remaining_length = len(encoded_bytes)
            if verbose and remaining_length % 1000 == 0:
                percentage = (remaining_length / total_length) * 100
                print(f"\rProcessing: {percentage:6.1f}% [{remaining_length}/{total_length}]", end="", flush=True)

            try :
                gamma_val = next(gamme_gen)  # Get the next gamma decoded value
            except StopIteration:
                raise ValueError("Invalid delta encoding: no more gamma values to decode.")
            
            remainder_size = gamma_val - 1  # The size of the remainder bits

            if len(encoded_bytes) < remainder_size:
                raise ValueError("Invalid delta encoding: stream cut off during value remainder extraction.")
            
            # 4. Slice out the final payload bits
            res = bitarray([1])
            res.extend(encoded_bytes[:remainder_size])
            
            # 5. Physically slice off the consumed value payload bits
            del encoded_bytes[:remainder_size]

            yield ba2int(res)

    def varbyte_decode(self, encoded_bytes, verbose=False): #! use yield
        # ptr = 0
        total_length = len(encoded_bytes)
        # last_logged = 0

        current_byte = bitarray()  # Accumulates the 7-bit payloads for the current integer

        while len(encoded_bytes) > 0:

            if len(encoded_bytes) < 8:
                raise ValueError("Invalid varbyte encoding: stream cut off during byte extraction.")

            remaining_length = len(encoded_bytes)
            if verbose and remaining_length % 1000 == 0:
                percentage = (remaining_length / total_length) * 100
                print(f"\rProcessing: {percentage:6.1f}% [{remaining_length}/{total_length}]", end="", flush=True)

            # 1. Read the flag bit and the payload chunks
            first_bit = encoded_bytes[0]
            current_byte.extend(encoded_bytes[1:8])

            # 2. Chop off the 8 bits we just processed
            del encoded_bytes[:8]

            # 3. If the flag bit is 1, the integer payload assembly is complete
            if first_bit == 1:
                yield ba2int(current_byte)
                current_byte.clear()

    def pfor_delta_encode_bit(self, d_gaps, k=8, extra_strategy=None): #! compression given the full d-gap list returns the compressed d-gap list and the outliers
        if not d_gaps:
            raise ValueError("Input d_gaps list is empty.")
    
        min_ = min(d_gaps)
        outliers = []
        
        # Initialize an empty bit array
        compressed_bits = bitarray()
        
        escape_value = (1 << k) - 1
        # Create a k-bit array of all 1s for the escape token
        escape_token = bitarray('1' * k) 
    
        for val in d_gaps:
            var = val - min_
            
            if var >= escape_value:
                outliers.append(extra_strategy(var))
                compressed_bits.extend(escape_token)
            else:
                # int2ba converts an integer directly to a bitarray of length k
                compressed_bits.extend(int2ba(var, length=k))
                
        # To get raw bytes for saving to a file or transmitting:
        # return compressed_bits.tobytes(), outliers, min_
        return compressed_bits, outliers, min_

    def pfor_delta_decode_bit(self, compressed_bits, outliers, min_val, k=8, decode_fallback=None):

        decoded = []
        # outlier_index = 0
        escape_value = (1 << k) - 1
        
        outlier_stream = iter(outliers)

        # Move through the bitarray in steps of k bits
        for i in range(0, len(compressed_bits), k):
            # Extract a k-bit chunk
            bit_slice = compressed_bits[i : i + k]
            
            # Guard clause: Ignore trailing padding bits that don't make a full k-bit chunk
            if len(bit_slice) < k:
                break
                
            # Convert the k bits directly to an integer
            val = ba2int(bit_slice)
            
            if val == escape_value:
                outlier_item = next(outlier_stream)
            
                # If the outlier is already an int ("None" strategy), use it directly
                if isinstance(outlier_item, int):
                    outlier_value = outlier_item
                else:
                    outlier_gen = decode_fallback(outlier_item)
                    outlier_value = next(outlier_gen)
                    
                decoded.append(outlier_value + min_val)
                # outlier_index += 1
            else:
                # Standard value, add the baseline
                decoded.append(val + min_val)
                
        return decoded 
    
    def handle_pfor_delta(self, is_decompressing=False):
        
        comp_methods_map = { 
            "None" : (lambda x: x, lambda x: x),
            "gamma": (self.gamma_encode, self.gamma_decode),
            "delta": (self.delta_encode, self.delta_decode),
            "varbyte": (self.varbyte_encode, self.varbyte_decode)
        }

        strategy = self.pfor_delta_extra_strategy
        encode_fallback, decode_fallback = comp_methods_map[strategy]

        for term, posting_list in self.index_obj.index.items():

            # ----------------------------------------
            # DECOMPRESSION LOGIC
            # ----------------------------------------
            if is_decompressing:
                full_decoded_list = []
                
                for block in posting_list:
                    # Detect PForDelta: It's always a tuple of 3 items
                    if isinstance(block, tuple):
                        compressed_bits, outliers, min_val = block
                        decoded_block = self.pfor_delta_decode_bit(
                            compressed_bits, outliers, min_val, k=self.pfor_delta_k,  decode_fallback=decode_fallback
                        )
                        full_decoded_list.extend(decoded_block)
                    
                    # Detect Fallback: It's just a raw bitarray/bytes object
                    else:
                        decoded_block = list(decode_fallback(block))
                        full_decoded_list.extend(decoded_block)
                
                self.index_obj.index[term] = full_decoded_list
                continue

            # ----------------------------------------
            # COMPRESSION LOGIC
            # ----------------------------------------
            d_gap_blocks = [posting_list[i : i + self.pfor_delta_block_size] for i in range(0, len(posting_list), self.pfor_delta_block_size)]
            compressed_term_data = []

            for block in d_gap_blocks:
                # Handle leftover block with fallback
                if len(block) < self.pfor_delta_block_size:
                    if strategy == "None":
                        # Store the raw, uncompressed list of integers directly
                        compressed_term_data.append(block)
                    else :
                        fallback_bits = bitarray()
                        for val in block:
                            # print(f"Encoding fallback value: {val} for term: {term} using strategy: {strategy} block length: {len(block)}")
                            fallback_bits.extend(encode_fallback(val))
                        
                        # Store ONLY the raw bitarray
                        compressed_term_data.append(fallback_bits)
                    continue

                # Handle full block with PForDelta
                # print(f"Encoding full block for term: {term} with PForDelta, block size: {len(block)}")
                pfor_tuple = self.pfor_delta_encode_bit(block, self.pfor_delta_k, extra_strategy=encode_fallback)
                
                # Store ONLY the 3-element tuple
                compressed_term_data.append(pfor_tuple)

            self.index_obj.index[term] = compressed_term_data
    
    def no_decompression(self, encoded_bytes, verbose=False): 
        # ptr = 0
        total_length = len(encoded_bytes)
        while len(encoded_bytes) > 0:

            remaining_length = len(encoded_bytes)
            if verbose and remaining_length % 1000 == 0:
                percentage = (remaining_length / total_length) * 100
                print(f"\rProcessing: {percentage:6.1f}% [{remaining_length}/{total_length}]", end="", flush=True)

            number = ba2int(encoded_bytes[:32])  # Assuming 32 bits for each integer
            del encoded_bytes[:32]  # Remove the processed bits
            yield number