from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    while True:
        user_input = input("Enter question: ")
        results = generator.chat_completion(
            [[{"role":"user","content":user_input}]],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"\033[1;31;40m{results[0]['generation']['content']}\033[0m\n")

if __name__ == "__main__":
    fire.Fire(main)
