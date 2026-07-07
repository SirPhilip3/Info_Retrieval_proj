import struct
from bitarray import bitarray
from bitarray.util import int2ba, ba2int

COMPRESSION_METHODS_ENC = {
    "None": 0,
    "gamma": 1,
    "delta": 2,
    "varbyte": 3,
    "pfor_delta": 4
}

COMPRESSION_METHODS_DEC = {
    0: "None",
    1: "gamma",
    2: "delta",
    3: "varbyte",
    4: "pfor_delta"
}

PFOR_DELTA_EXTRA_STRATEGY_ENC = { # strategy for the last block in PForDelta encoding and outliers handling
    0: "None",
    1: "gamma",
    2: "delta",
    3: "varbyte",
}

PFOR_DELTA_EXTRA_STRATEGY_DEC = {
    "None": 0,
    "gamma": 1,
    "delta": 2,
    "varbyte": 3,
}

MAX_UINT32 = 2**32 - 1

class IndexStorage: 
    def __init__(self, index_obj, file_path):
        self.index_obj = index_obj
        self.file_path = file_path
        # pass
    
    def clear_storage(self):
        self.file_path = None

    def set_file_path(self, file_path):
        self.file_path = file_path
    
    def build_header(self):
        if self.index_obj.compression_method == "pfor_delta":
            extra_strategy_code = PFOR_DELTA_EXTRA_STRATEGY_DEC[self.index_obj.compressor.pfor_delta_extra_strategy]
            header = struct.pack('<IBBIII', 
                                 COMPRESSION_METHODS_ENC[self.index_obj.compression_method], 
                                 self.index_obj.removed_tf,
                                 self.index_obj.d_gap_index_, 
                                 extra_strategy_code, 
                                 self.index_obj.compressor.pfor_delta_k,
                                 self.index_obj.compressor.pfor_delta_block_size)
            return header
        compression_code = COMPRESSION_METHODS_ENC[self.index_obj.compression_method]
        header = struct.pack('<IBB', compression_code, self.index_obj.removed_tf, self.index_obj.d_gap_index_)
        return header
    
    def write_to_disk(self):
        header = self.build_header()
        
        with open(self.file_path, 'wb') as f:
            
            f.write(header)

            combined_bits = bitarray()

            for term, posting_list in self.index_obj.index.items():

                combined_bits.clear()  # Reset combined_bits for each term

                term_bytes = term.encode('utf-8')
                term_len = len(term_bytes)
                f.write(struct.pack('<I', term_len))
                f.write(struct.pack(f'<{term_len}s', term_bytes))
                
                if self.index_obj.removed_tf:

                    for doc_id in posting_list:
                        # print(doc_id.__class__.__name__)

                        if self.index_obj.compression_method == "pfor_delta":
                            if isinstance(doc_id, tuple) and len(doc_id) == 3:
                                # print(f"Compressing term: {term}, doc_id: {doc_id}")

                                combined_bits.extend(int2ba(0, length=8))  
                                combined_bits.extend(int2ba(doc_id[2], length=32))  
                                combined_bits.extend(doc_id[0])
                                for item in doc_id[1]:
                                    if isinstance(item, int):
                                        print(f"Compressing term: {term}, doc_id: {doc_id}, item: {item} (as int)")
                                        bits_item = int2ba(item, length=32)
                                        combined_bits.extend(bits_item)
                                    else:
                                        combined_bits.extend(item)
                            elif isinstance(doc_id, list):

                                
                                print(f"Compressing term: {term}, doc_id: {doc_id} (as list)")
                                combined_bits.extend(int2ba(1, length=8)) 
                                for item in doc_id:
                                    if isinstance(item, int):
                                        # print(f"Compressing term: {term}, doc_id: {doc_id}, item: {item} (as int)")
                                        bits_item = int2ba(item, length=32)
                                        combined_bits.extend(bits_item)
                                    else:
                                        combined_bits.extend(item) 
                            else:
                                # print(f"Compressing term: {term}, doc_id: {doc_id}")
                                combined_bits.extend(int2ba(1, length=8))
                                combined_bits.extend(doc_id)
                            
                            continue  # Skip the rest of the loop since we've already processed the PForDelta blocks

                        if isinstance(doc_id, int):
                            combined_bits.extend(int2ba(doc_id, length=32))  # Convert integer to bitarray of fixed length
                        else:
                            # print(f"Compressing term: {term}, doc_id: {doc_id}")
                            combined_bits.extend(doc_id) 
                    
                else: 
                    for doc_id, tf in posting_list:
                        if isinstance(doc_id, int):
                            combined_bits.extend(int2ba(doc_id, length=32))
                            combined_bits.extend(int2ba(tf, length=32))  # Convert integer to bitarray of fixed length
                        else:   
                            # print(f"Compressing term: {term}, doc_id: {doc_id}, tf: {tf}")
                            combined_bits.extend(doc_id)
                            combined_bits.extend(tf)

                bit_length = len(combined_bits)

                if bit_length > MAX_UINT32:
                    raise ValueError(f"Posting list for term '{term}' is too large to be represented in 4 bytes. Bit length: {bit_length} bits.")

                f.write(struct.pack('<I', bit_length))

                
                combined_bits.tofile(f)

    def read_from_disk(self):

        if self.index_obj.index != {}:
            raise ValueError("Index object is not empty. Cannot read from disk into a non-empty index.")

        with open(self.file_path, 'rb') as f:

            header = f.read(6)  
            compression_code, removed_tf_flag, d_gap_index_flag = struct.unpack('<IBB', header)
            
            print(f"Loading index with compression code: {compression_code}, removed_tf_flag: {removed_tf_flag}, d_gap_index_flag: {d_gap_index_flag}")
            
            detected_method = COMPRESSION_METHODS_DEC[compression_code]
            
            self.index_obj.compression_method = "None" 
            self.index_obj.removed_tf = bool(removed_tf_flag)
            self.index_obj.d_gap_index_ = bool(d_gap_index_flag)

            self.decompress_index_from_file(f, detected_method) 

    
    def decompress_index_from_file(self, index_data, decompression_method):

        decomp_methods_map = { # also used for extra decompression strategy in PForDelta
            "None" : self.index_obj.compressor.no_decompression,
            "gamma": self.index_obj.compressor.gamma_decode,
            "delta": self.index_obj.compressor.delta_decode,
            "varbyte": self.index_obj.compressor.varbyte_decode,
            "pfor_delta": None # we do it in place here
        }

        if decompression_method not in decomp_methods_map:
            raise ValueError(f"Decompression method '{decompression_method}' is not supported. Allowed methods: {list(decomp_methods_map.keys())}")
        
        decomp_func = decomp_methods_map[decompression_method]

        if decompression_method == "pfor_delta":
            extra_strategy_code = struct.unpack('<B', index_data.read(1))[0]
            self.index_obj.compressor.pfor_delta_extra_strategy = PFOR_DELTA_EXTRA_STRATEGY_ENC[extra_strategy_code]
            self.index_obj.compressor.pfor_delta_k = struct.unpack('<I', index_data.read(4))[0]
            self.index_obj.compressor.pfor_delta_block_size = struct.unpack('<I', index_data.read(4))[0]

        while True:
            term_len_bytes = index_data.read(4)
            if not term_len_bytes:
                break # Clean End of File reached

            term_len = struct.unpack('<I', term_len_bytes)[0]
            term_bytes = index_data.read(term_len)
            if not term_bytes:
                break # Clean End of File reached
            
            print(f"Reading term of length {term_len} bytes: {term_bytes}")
            term = term_bytes.decode('utf-8') 

            size_bytes = index_data.read(4)
            if not size_bytes:
                break # Clean End of File reached
                
            pl_size = struct.unpack('<I', size_bytes)[0] 
            
            # print(f"Decompressing term: {moch_term}, Posting list size: {pl_size}, Reading: {index_data.tell()} bytes")
            self.index_obj.index[term] = []

            byte_length = (pl_size + 7) // 8
        
            
            bits = bitarray()
            bits.fromfile(index_data, byte_length)

            bits = bits[:pl_size] 

            if decompression_method == "pfor_delta":
                extra_decomp_function = decomp_methods_map[self.index_obj.compressor.pfor_delta_extra_strategy]
                while len(bits) > 0:

                    
                    if len(bits) < 8:  # Safety check for trailing padding bits
                        break
                        
                    block_type_indicator = ba2int(bits[:8])  # Read the 8-bit indicator
                    # print(f"Block type indicator for term '{term}': {block_type_indicator}, Remaining bits: {len(bits)}")
                    del bits[:8]


                    if block_type_indicator == 1:  # Last block
                        # Handle the last block with the extra strategy
                        # print(f"Processing last block for term: {term}, Remaining bits: {len(bits)}")
                        final_doc_ids = extra_decomp_function(bits)
                        for doc_id in final_doc_ids:
                            self.index_obj.index[term].append(doc_id)  # Assuming doc_id is already a bitarray
                        break

                    min_value = ba2int(bits[:32])
                    del bits[:32]
                    
                    k = self.index_obj.compressor.pfor_delta_k
                    
                    block_size = self.index_obj.compressor.pfor_delta_block_size * k

                    # print(f"Processing block for term: {term}, Remaining bits: {len(bits)}, Block size: {block_size}, k: {k}, min_value: {min_value}")

                    block = bits[:block_size]
                    del bits[:block_size] # remove the processed block from bits
                    
                    chunks = [block[i : i + k] for i in range(0, len(block), k)]

                    outlier_count = 0
                    outlier_positions = []
                    outliers = []
                    for i, chunk in enumerate(chunks):
                        if chunk.count(1) == self.index_obj.compressor.pfor_delta_k:
                            outlier_count += 1
                            outlier_positions.append(i)
                        
                        outliers.append(ba2int(chunk)+min_value)  # Convert the chunk to an integer to ensure it's valid


                    outlier_generator = extra_decomp_function(bits)
                    # count = 0
                    if outlier_count > 0:
                        for count in range(outlier_count):
                            el = next(outlier_generator)
                            outliers[outlier_positions[count]] = el + min_value
                    
                    self.index_obj.index[term].extend(outliers)
                
                continue  # Skip the rest of the loop since we've already processed the PForDelta blocks
            
            if self.index_obj.removed_tf:
                for doc_id in decomp_func(bits):
                    self.index_obj.index[term].append(doc_id)
            else:
                iterator = iter(decomp_func(bits)) 

                for doc_id, tf in zip(iterator, iterator):
                    self.index_obj.index[term].append((doc_id, tf))
                