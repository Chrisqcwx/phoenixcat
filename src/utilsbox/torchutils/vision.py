from typing import Callable, Optional


def augment_images_iterator(
    initial_transform: Optional[Callable] = None,
    add_origin_image: bool = True,
    augment: Optional[Callable] = None,
    augment_times: int = 0,
):
    """Return a function for image augmentation.

    Args:
        initial_transform (Optional[Callable], optional): The first transformation to perform. Defaults to None.
        add_origin_image (bool, optional): Whether to return the original image. Defaults to True.
        augment (Optional[Callable], optional): The augmentation to perform. Defaults to None.
        augment_times (int, optional): Times for augmentation to repeat. Defaults to 0.
    """

    def fn(image):
        if initial_transform is not None:
            image = initial_transform(image)

        if add_origin_image:
            yield image

        if augment is not None:
            for i in range(augment_times):
                yield augment(image)

    return fn
