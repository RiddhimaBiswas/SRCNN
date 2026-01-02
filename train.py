from src.model import create_cnn

position = [...]  # SAME array

final_loss = create_cnn(
    position=position,
    input_size=127,
    batch_size=2,
    epochs=100,
    image_dir="data/input",
    target_dir="data/target",
    analyze=True
)

print("Final Training Loss:", final_loss)
