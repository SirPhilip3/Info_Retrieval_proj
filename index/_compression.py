from bitarray import bitarray
from bitarray.util import ba2int, int2ba

class IndexCompressor:

    def __init__(self, index_obj=None):
        self.index_obj = index_obj

    def apply_index_transformation(self, transform_func, is_decompressing=False):
        for term, posting_list in self.index_obj.index.items():
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
        return '1' * number + '0'

    def gamma_encode(self, number):
        bin_num = bin(number)[2:]

        rem = bin_num[1:]  # remove leading '1'
        length = len(bin_num)
        una = self.transform_unary(length - 1)

        # print(f"Gamma encoding number: {number}, Binary: {bin_num}, Unary: {una}, Remainder: {rem}")

        return bitarray(una + rem) #! should be better reimplemented with bitarray directly ALSO YIELD

    def delta_encode(self, number):
        bin_num = bin(number)[2:]

        rem = bin_num[1:]  # remove leading '1'
        length = len(bin_num)
        gamma_length = self.gamma_encode(length)
        
        return bitarray(gamma_length + rem) #! should be better reimplemented with bitarray directly ALSO YIELD

    def varbyte_encode(self, number):
        bin_num = bin(number)[2:] # remove '0b' prefix

        # print(f"Encoding number: {number}, Binary representation: {bin_num}")

        cl = ((len(bin_num) + 6) // 7) * 7
        bin_num = bin_num.zfill(cl)  # Pad with leading zeros to make its length a multiple of 8
        spits = [bin_num[i:i+7] for i in range(0, len(bin_num), 7)]
        
        # print(f"Number: {number}, Binary: {bin_num}, Spits: {spits}")
        
        last_byte = '1' + spits[-1]
        other_bytes = ['0' + s for s in spits[:-1]]
        
        # print(f"Last byte: {last_byte}, Other bytes: {other_bytes}")
        
        res = ''.join(other_bytes + [last_byte])

        return bitarray(res) #! should be better reimplemented with bitarray directly ALSO YIELD

    def gamma_decode(self, encoded_bytes): #! use yield

        ptr = 0
        total_length = len(encoded_bytes)

        # if self.removed_tf:
        while ptr < total_length:
            to_read = 0
            n = total_length - ptr
            
            # 1. Find the first 0 (the end of the unary prefix)
            for i in range(ptr, total_length):
                if encoded_bytes[i] == 0:
                    to_read = i - ptr
                    break
            else:
                # Break out if no 0 is found to prevent an infinite loop on corrupt data
                raise ValueError("Invalid gamma encoding: no terminating 0 found in unary prefix.")
                
            
            # 2. Remove the unary prefix entirely (the 1s and the terminating 0)
            # for _ in range(to_read + 1):
            #     encoded_bytes.pop(0) #! bad performance maybe bc it needs to shift the whole array every time, maybe we can just keep an index pointer instead of popping
            
            # 3. Create a new bitarray starting with the implicit leading 1
            
            remainder_start = ptr + to_read + 1
            remainder_end = remainder_start + to_read

            res = bitarray([1])
            res.extend(encoded_bytes[remainder_start:remainder_end])  # Add the remainder bits
            
            # 4. Physically pop the exact number of remainder bits needed
            # for _ in range(to_read):
            #     if len(encoded_bytes) > 0: #! could be useless
            #         res.append(encoded_bytes.pop(0))
            
            ptr = remainder_end  # Move the pointer past the remainder bits

            yield ba2int(res)

    def delta_decode(self, encoded_bytes): #! use yield
        #! since we are use yield we cant call gamma decode directly , think about this for reuse of code
        while len(encoded_bytes) > 0:
            to_read = 0
            n = len(encoded_bytes)

            # 1. Find the first 0 (the end of the unary prefix)
            for i in range(n):
                if encoded_bytes[i] == 0:
                    to_read = i
                    break
            else:
                # Break out if no 0 is found to prevent an infinite loop on corrupt data
                raise ValueError("Invalid delta encoding: no terminating 0 found in unary prefix.")
                
            # 2. Remove the unary prefix entirely (the 1s and the terminating 0)
            for _ in range(to_read + 1):
                encoded_bytes.pop(0)
            
            # 3. Create a new bitarray starting with the implicit leading 1
            rem_1 = bitarray([1])
            
            # 4. Physically pop the exact number of remainder bits needed
            for _ in range(to_read):
                if len(encoded_bytes) > 0: #! could be useless
                    rem_1.append(encoded_bytes.pop(0))
            
            remainder_size = ba2int(rem_1) - 1  

            res = bitarray([1])
            
            # Physically pop the remainder bits needed for the actual value
            for _ in range(remainder_size):
                if len(encoded_bytes) > 0:
                    res.append(encoded_bytes.pop(0))
                else:
                    raise ValueError("Invalid delta encoding: stream cut off during remainder extraction.")
                    
            yield ba2int(res)

    def varbyte_decode(self, encoded_bytes): #! use yield

        while len(encoded_bytes) > 0:
            first_bit = encoded_bytes.pop(0)
            current_byte = bitarray()  
            while first_bit == 0:
                # Read the next 7 bits for the current byte
                for _ in range(7):
                    if len(encoded_bytes) > 0:
                        current_byte.append(encoded_bytes.pop(0))
                    else:
                        raise ValueError("Invalid varbyte encoding: stream cut off during byte extraction.")
                
                if len(encoded_bytes) > 0:
                    first_bit = encoded_bytes.pop(0)
                else:
                    raise ValueError("Invalid varbyte encoding: stream cut off after last byte.")
            
            for _ in range(7):
                if len(encoded_bytes) > 0:
                    current_byte.append(encoded_bytes.pop(0))
                else:
                    raise ValueError("Invalid varbyte encoding: stream cut off during last byte extraction.")
                
            yield ba2int(current_byte)
            current_byte.clear()

        # encoded_str = encoded_bytes.to01()
        # com_string = ''
        # for i in range(0, len(encoded_str), 8):
        #     byte = encoded_str[i:i+8]
        #     if byte[0] == '0':  # last byte
        #         com_string += byte[1:]
        #     else:
        #         com_string += byte[1:]
        #         print(f"Varbyte decoding string: {encoded_str}, Byte: {byte}, Result so far: {int(com_string, 2)}")
        #         yield com_string
        #         com_string = ''

        # return com_string

    def pfor_delta_encode(self, d_gaps, k=4): #! compression given the full d-gap list returns the compressed d-gap list and the outliers
        min_ = min(d_gaps)

        outliers = []

        for i in range(len(d_gaps)):
            var = d_gaps[i] - min_
            if var >= 2**k-1:
                outliers.append(bin(var)[2:])
                d_gaps[i] = "1"*k
            else:
                d_gaps[i] = bin(var)[2:].zfill(k)
        
        # print(f"PForDelta encoding: D-gaps: {d_gaps}, Outliers: {outliers}")
        return d_gaps, outliers, min_

    def pfor_delta_decode(self, d_gap_array, outliers, min, k=4):
        decoded = []
        outlier_index = 0
        for d_gap in d_gap_array:
            if d_gap == "1"*k:
                decoded.append(int(outliers[outlier_index], 2) + min)
                outlier_index += 1
            else:
                decoded.append(int(d_gap, 2) + min)
        
        return decoded
