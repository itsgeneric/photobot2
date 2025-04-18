import os
import cv2
import face_recognition
import discord
from sklearn.cluster import DBSCAN
import asyncio
import pickle
import warnings

DISCORD_PUBLIC_KEY = "YOUR_API_KEY_HERE"


class FaceClusterBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_name = None
        self.known_people = {}
        self.load_known_people()
        self.tree = discord.app_commands.CommandTree(self)

    def save_known_people(self):
        with open("known_people.pkl", "wb") as file:
            pickle.dump(self.known_people, file)

    def load_known_people(self):
        if os.path.exists("known_people.pkl"):
            with open("known_people.pkl", "rb") as file:
                self.known_people = pickle.load(file)

    async def process_face_clusters(self, channel, images):
        try:
            all_face_encodings = []
            for img_path in images:
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img, model = "hog")
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                all_face_encodings.extend(face_encodings)

            if not all_face_encodings:
                await channel.send("❌ No faces detected!")
                return {}

            clustering = DBSCAN(eps = 0.5, min_samples = 1).fit(all_face_encodings)
            clusters = {label: [] for label in set(clustering.labels_)}
            for encoding, label in zip(all_face_encodings, clustering.labels_):
                clusters[label].append(encoding)

            return clusters
        except Exception as e:
            await channel.send(f"❌ Error during face clustering: {e}")
            return {}

    async def convert_user_id_to_numeric(self, user_id: str):
        try:
            if user_id.startswith("<@") and user_id.endswith(">"):
                numeric_id = int(user_id.strip("<@!>"))
            else:
                numeric_id = int(user_id)

            # Validate user existence
            user = await self.fetch_user(numeric_id)
            if user:
                return numeric_id
            else:
                return None
        except Exception as e:
            print(f"Error converting user ID: {e}")
            return None

    def recognize_faces(self, face_encodings):
        recognized_users = set()
        used_encodings = set()  # Track which encodings have been matched

        # Sort faces by their match confidence
        face_matches = []
        for idx, face_encoding in enumerate(face_encodings):
            best_match = None
            best_confidence = float('inf')

            for user_id, known_encodings in self.known_people.items():
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = min(distances)

                if min_distance < best_confidence:
                    best_confidence = min_distance
                    best_match = user_id

            if best_match is not None:
                face_matches.append((idx, best_match, best_confidence))

        # Sort matches by confidence (lower distance = better match)
        face_matches.sort(key = lambda x: x[2])

        # Process matches ensuring each face encoding is only used once
        for idx, user_id, confidence in face_matches:
            if idx not in used_encodings and confidence < 0.5:  # Stricter threshold
                recognized_users.add(user_id)
                used_encodings.add(idx)

        return recognized_users

    def associate_face_with_user(self, user_id, face_encoding):
        if user_id in self.known_people:
            self.known_people[user_id].append(face_encoding)
        else:
            self.known_people[user_id] = [face_encoding]
        self.save_known_people()

    async def shutdown(self):
        print("Shutting down bot...")
        self.save_known_people()
        await self.close()
        print("Bot has been shut down.")


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
bot = FaceClusterBot(intents = intents)


@bot.event
async def on_ready():
    if bot.guilds:
        bot.server_name = bot.guilds[0].name
    print(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    await bot.tree.sync()
    print("Slash commands synced.")


@bot.tree.command(name = "associate", description = "Associate a user's face with their account.")
async def associate(interaction: discord.Interaction, user: discord.User):
    await interaction.response.send_message(
        f"📸 {interaction.user.mention}, please upload a photo to associate with {user.mention}."
    )

    def check(m):
        return m.author == interaction.user and m.attachments

    try:
        msg = await bot.wait_for("message", check = check, timeout = 60.0)
        attachment = msg.attachments[0]

        if not attachment.filename.lower().endswith(("png", "jpg", "jpeg")):
            await interaction.followup.send("⚠️ Invalid file type. Please upload a valid image.")
            return

        file_path = f"./downloads/{attachment.filename}"
        await attachment.save(file_path)

        img = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img, model = "hog")
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_encodings:
            await interaction.followup.send(
                f"❌ No faces detected in the uploaded image. Please try again, {interaction.user.mention}."
            )
            return
        face_encoding = face_encodings[0]
        bot.associate_face_with_user(user.id, face_encoding)

        await interaction.followup.send(f"✅ Successfully associated the uploaded face with {user.mention}.")

    except asyncio.TimeoutError:
        await interaction.followup.send(
            f"⏳ Timeout! You didn't upload an image in time, {interaction.user.mention}."
        )
    except Exception as e:
        await interaction.followup.send(f"❌ An error occurred: {e}")


@bot.tree.command(name = "analyze", description = "Analyze a group photo and recognize faces.")
async def analyze(interaction: discord.Interaction):
    await interaction.response.send_message("📸 Please upload a photo to process.")

    def check(m):
        return m.author == interaction.user and m.attachments

    try:
        msg = await bot.wait_for("message", check = check, timeout = 60.0)
        attachment = msg.attachments[0]

        if not attachment.filename.lower().endswith(("png", "jpg", "jpeg")):
            await interaction.followup.send("⚠️ Invalid file type. Please upload a valid image.")
            return

        file_path = f"./downloads/{attachment.filename}"
        await attachment.save(file_path)
        img = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img, model = "hog")
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_encodings:
            await interaction.followup.send("❌ No faces detected in the uploaded image.")
            return

        recognized_users = bot.recognize_faces(face_encodings)

        if recognized_users:
            await interaction.followup.send(
                f"✅ Recognized {len(recognized_users)} faces. Sending DMs to associated users.")
            for user_id in recognized_users:
                user = await bot.fetch_user(user_id)
                if user:
                    try:
                        with open(file_path, "rb") as img_file:
                            image_data = discord.File(img_file, filename = attachment.filename)
                            await user.send(
                                f"📢 Your face was recognized in a photo in {bot.server_name} !",
                                file = image_data
                            )
                    except discord.Forbidden:
                        await interaction.followup.send(f"⚠️ Unable to DM user {user.name} (ID: {user_id}).")
                else:
                    await interaction.followup.send(f"⚠️ Could not fetch user with ID: {user_id}.")
        else:
            await interaction.followup.send("⚠️ No recognized faces in the image.")

    except asyncio.TimeoutError:
        await interaction.followup.send("⏳ Timeout! You didn't upload an image in time.")
    except Exception as e:
        await interaction.followup.send(f"❌ An error occurred: {e}")


@bot.tree.command(name = "reset", description = "Reset all face associations.")
async def reset(interaction: discord.Interaction):
    bot.known_people = {}
    bot.save_known_people()
    await interaction.response.send_message("✅ All face associations have been reset.")


if not os.path.exists('./downloads'):
    os.makedirs('./downloads')


def run_bot():
    async def main():
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        warnings.filterwarnings("ignore", category = ResourceWarning)

        async with bot:
            try:
                await bot.start('BOT_START_KEY')
            except Exception as e:
                print(f"Error during bot startup: {e}")
            finally:
                try:
                    await bot.close()
                except Exception:
                    pass

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(lambda loop, context: None)
        asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt:
        print("Bot shut down manually.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            loop = asyncio.get_event_loop()
            loop.close()
        except Exception:
            pass


run_bot()